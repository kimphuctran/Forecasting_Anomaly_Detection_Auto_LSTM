#!/usr/bin/env python3
"""Official-source SSL controls on independent UCR train/test datasets."""
import argparse,json,time,traceback
from pathlib import Path
import numpy as np
from bench_neuro_ssl import eval_probe,ts2vec,timemae,moment,seed_everything
from MESA_SSL.mesa_canonical import load_data

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[209,210,211])
    p.add_argument('--epochs',type=int,default=35)
    p.add_argument('--out',default='results/baselines')
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    tr,te,yt,yv,classes=load_data(a.dataset)
    rows=[]
    for seed in a.seeds:
        for method in ['Raw','TS2Vec','TimeMAE','MOMENT-1-small']:
            start=time.monotonic()
            try:
                if method=='Raw':
                    x,z=tr.reshape(len(tr),-1),te.reshape(len(te),-1)
                    meta={'implementation':'Train-split normalized Raw + LR'}
                elif method=='TS2Vec':x,z,meta=ts2vec(tr,te,seed,a.epochs)
                elif method=='TimeMAE':x,z,meta=timemae(tr,te,seed,a.epochs)
                else:x,z,meta=moment(tr,te,seed,a.epochs)
                if not np.isfinite(x).all() or not np.isfinite(z).all():
                    raise ValueError('Nonfinite representation')
                scores=eval_probe(x,z,yt,yv,seed,budget=5)
                row={'dataset':a.dataset,'seed':seed,'method':method,'status':'EXECUTED',
                     'train_n':len(tr),'test_n':len(te),'epochs':0 if method in ('Raw','MOMENT-1-small') else a.epochs,
                     'seconds':round(time.monotonic()-start,2),**scores,**meta}
            except Exception as e:
                row={'dataset':a.dataset,'seed':seed,'method':method,'status':'BLOCKED',
                     'error':str(e)[:600],'traceback':traceback.format_exc()[-900:]}
            rows.append(row);print('RESULT_JSON='+json.dumps(row,allow_nan=False),flush=True)
            (out/'results.json').write_text(json.dumps(rows,indent=2))
    if any(r['status']!='EXECUTED' for r in rows):
        raise RuntimeError('Incomplete SSL controls, inspect results.json')

if __name__=='__main__':main()
