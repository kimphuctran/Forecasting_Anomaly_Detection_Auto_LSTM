#!/usr/bin/env python3
"""Identical logistic-regression readout for feature-level UCR comparison."""
import argparse,json,time
from pathlib import Path
import numpy as np
from aeon.transformations.collection.convolution_based import MiniRocket
from bench_neuro_ssl import eval_probe
from mesa_mr_experiment import train_encoder,encode
from MESA_SSL.mesa_canonical import load_data,separate

def representations(train,test,seed):
    model=MiniRocket(n_kernels=1008,random_state=seed,n_jobs=1)
    return (np.asarray(model.fit_transform(train.transpose(0,2,1))),
            np.asarray(model.transform(test.transpose(0,2,1))))

def run(args):
    tr,te,y,z,cls=load_data(args.dataset)
    cr,cn=separate(tr);ce,cm=separate(te)
    rows=[];out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    for seed in args.seeds:
        raw_a,raw_b=representations(tr,te,seed)
        can_a,can_b=representations(cr,ce,seed)
        model,_=train_encoder(cr,seed,args.epochs,mechanism=True)
        emb_a,emb_b=encode(model,cr),encode(model,ce)
        methods={
          'MiniRocket+LR':(raw_a,raw_b),
          'CanonicalMiniRocket+LR':(can_a,can_b),
          'CanonicalMiniRocket+Concepts+LR':(np.c_[can_a,cn],np.c_[can_b,cm]),
          'MESA-MR+LR':(emb_a,emb_b),
          'CanonicalMiniRocket+MESA-MR+LR':(np.c_[can_a,cn,emb_a],np.c_[can_b,cm,emb_b])}
        for name,(a,b) in methods.items():
            score=eval_probe(a,b,y,z,seed,budget=5)
            rec={'dataset':args.dataset,'seed':seed,'method':name,
                 'status':'EXECUTED','macro_f1':score['macro_f1'],'accuracy':score['accuracy'],
                 'n_labeled':score['n_labeled'],'epochs':args.epochs if 'MESA-MR' in name else 0,
                 'protocol':'Same fixed LogisticRegression default with train-only normalization'}
            rows.append(rec);print('RESULT_JSON='+json.dumps(rec),flush=True)
        (out/'results.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[209,210,211])
    p.add_argument('--epochs',type=int,default=35);p.add_argument('--out',default='results')
    args=p.parse_args();run(args)
