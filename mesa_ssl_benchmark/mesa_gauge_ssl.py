#!/usr/bin/env python3
"""Gauge-separated mechanism-conditioned SSL evaluation.

No labels from TEST enter fitting, normalization, SSL pretraining, or model
selection. Offset/trend interventions in evaluation are synthetic and
class-preserving by experimental construction, not claims about physical data.
"""
import argparse, json, time, traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from aeon.transformations.collection.convolution_based import MiniRocket

from bench_neuro_ssl import acquire, seed_everything
from mesa_mr_experiment import MESAEncoder, indices_for_seed


def affine_gauge(x, eps=1e-6):
    """Orthogonal residual and interpretable (mean, slope, residual RMS).
    x: (case, time, channel), works with NumPy arrays only.
    """
    t=np.linspace(-1,1,x.shape[1],dtype=np.float64)
    a=x.astype(np.float64)
    mu=a.mean(1,keepdims=True)
    slope=(a*t[None,:,None]).sum(1,keepdims=True)/(t*t).sum()
    residual=a-mu-t[None,:,None]*slope
    rms=np.sqrt((residual**2).mean(1,keepdims=True))
    q=residual/np.maximum(rms,eps)
    concepts=np.concatenate([mu[:,0,:],slope[:,0,:],np.log(np.maximum(rms[:,0,:],eps))],axis=1)
    return q.astype('float32'),concepts.astype('float32')


def harmonic_basis(length):
    t=np.linspace(-1.,1.,length,dtype='float64')
    b=np.stack([np.cos(2*np.pi*t),np.sin(2*np.pi*t)],axis=1)
    b=b-b.mean(axis=0,keepdims=True)
    b=b-t[:,None]*(t[:,None]*b).sum(axis=0,keepdims=True)/(t*t).sum()
    b=b/np.maximum(np.sqrt(np.mean(b*b,axis=0,keepdims=True)),1e-8)
    return b.astype('float32')


def analytic_shape_concepts(x,basis):
    return torch.einsum('ntc,tk->nck',x,basis)/x.shape[1]


def project_shape_torch(x):
    length=x.shape[1]
    t=torch.linspace(-1.,1.,length,dtype=x.dtype,device=x.device)
    mu=x.mean(1,keepdim=True)
    slope=(x*t[None,:,None]).sum(1,keepdim=True)/(t*t).sum()
    r=x-mu-t[None,:,None]*slope
    sigma=r.square().mean(1,keepdim=True).sqrt().clamp_min(1e-6)
    return r/sigma


def pretrain(train, seed, epochs, mechanism):
    seed_everything(seed)
    data=torch.as_tensor(train,dtype=torch.float32)
    net=MESAEncoder(data.shape[-1],width=48,representation_dim=96)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    basis=torch.tensor(harmonic_basis(data.shape[1]))
    losses=[]
    for epoch in range(epochs):
        batch_losses=[]
        for ids in torch.randperm(len(data)).split(64):
            x=data[ids]
            mask=torch.rand_like(x)<.16
            corrupted=torch.where(mask,torch.zeros_like(x),x+.035*torch.randn_like(x))
            z,c,reconstruction=net(corrupted)
            zn,_,_=net(x+.045*torch.randn_like(x))
            reconstruction_loss=F.smooth_l1_loss(reconstruction[mask],x[mask]) if mask.any() else F.smooth_l1_loss(reconstruction,x)
            contrast=1-F.cosine_similarity(z,zn,dim=-1).mean()
            variance=F.relu(.5-z.std(0,unbiased=False)).mean()
            loss=reconstruction_loss+.12*contrast+.02*variance
            if mechanism:
                target=analytic_shape_concepts(x,basis).flatten(1)
                alpha=.15*torch.randn(len(x),1,x.shape[2])
                beta=.15*torch.randn(len(x),1,x.shape[2])
                shifted=project_shape_torch(x+basis[None,:,0:1]*alpha+basis[None,:,1:2]*beta)
                _,c_clean,_=net(x)
                _,c_shift,_=net(shifted)
                shifted_target=analytic_shape_concepts(shifted,basis).flatten(1)
                loss=loss+1.2*F.smooth_l1_loss(c_clean,target)
                loss=loss+.7*F.smooth_l1_loss(c_shift-c_clean,shifted_target-target)
                loss=loss+.25*F.smooth_l1_loss(c_shift,shifted_target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5)
            opt.step()
            batch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(batch_losses)))
    return net.eval(),losses


def encode(net,data):
    chunks=[]
    with torch.no_grad():
        for batch in np.array_split(data,max(1,int(np.ceil(len(data)/64)))):
            z,c,_=net(torch.tensor(batch,dtype=torch.float32))
            chunks.append(np.concatenate((z.numpy(),c.numpy()),axis=1))
    return np.concatenate(chunks,axis=0)


def scale_pair(train, test):
    scaler=StandardScaler().fit(train)
    a=scaler.transform(train)
    b=scaler.transform(test)
    a=np.clip(a,-10,10);b=np.clip(b,-10,10)
    fac=1/np.sqrt(a.shape[1])
    return a*fac,b*fac,scaler


def kernel_score(train,train_labels,ids,blocks,weights):
    """Select candidate by exact TRAIN-only leave-one-out ridge classification."""
    classes=np.unique(train_labels)
    lab=train_labels[ids]
    Y=(lab[:,None]==classes[None,:]).astype(float)
    results=[]
    for w in weights:
        train_features=np.c_[w*blocks['latent'][0],blocks['shape'][0]]
        sampled=train_features[ids]
        K=sampled@sampled.T
        for reg in (.01,.1,1.,10.):
            inv=np.linalg.inv(K+reg*np.eye(len(ids)))
            alpha=inv@Y
            loo=Y-alpha/np.diag(inv)[:,None]
            score=f1_score(lab,classes[loo.argmax(axis=1)],average='macro')
            results.append((score,w,reg,alpha,sampled))
    score,w,reg,alpha,sampled=sorted(results,key=lambda q:(-q[0],q[1],q[2]))[0]
    return {'weight':float(w),'ridge':float(reg),'cv_macro_f1':float(score),'classes':classes,'alpha':alpha,'sampled':sampled}


def infer(model,latent,shape):
    feat=np.c_[model['weight']*latent,shape]
    # The score is a sum of contributions from learned and kernel-feature blocks.
    margin=feat@model['sampled'].T@model['alpha']
    return model['classes'][margin.argmax(axis=1)],margin


def generate_affine_shift(x, seed, strength):
    rng=np.random.default_rng(seed*931+int(strength*1000)+97)
    channels=x.shape[-1]
    a=rng.standard_normal((len(x),1,channels)).astype('float32')
    b=rng.standard_normal((len(x),1,channels)).astype('float32')
    t=np.linspace(-1.,1.,x.shape[1],dtype='float32')[None,:,None]
    return x+strength*(.7*a+.7*b*t)


def verify_gauge():
    rng=np.random.default_rng(31)
    x=rng.normal(size=(12,100,3)).astype('float32')
    y=generate_affine_shift(x,51,1.6)
    g0,c0=affine_gauge(x);g1,c1=affine_gauge(y)
    assert np.max(np.abs(g0-g1))<5e-5,'Gauge invariance failed'
    assert np.max(np.abs(c0[:,-3:]-c1[:,-3:]))<5e-5,'RMS origin changed'
    basis=torch.tensor(harmonic_basis(100));xt=torch.tensor(g0)
    assert torch.max(torch.abs(xt.mean(1))).item()<1e-5
    r=project_shape_torch(torch.tensor(x))
    assert torch.max(torch.abs(r-torch.tensor(g0))).item()<1e-4
    return True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True,choices=['GunPoint','ItalyPowerDemand','BasicMotions','ArrowHead'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[301,302,303])
    parser.add_argument('--epochs',type=int,default=35)
    parser.add_argument('--out',default='results/mesa_gauge')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    verify_gauge()
    output=Path(args.out);output.mkdir(parents=True,exist_ok=True)
    tr,te,yt,yv,classes=acquire(args.dataset)
    x_train,_=affine_gauge(tr);x_test,_=affine_gauge(te)
    allrows=[]
    for seed in args.seeds:
        started=time.time()
        ids=indices_for_seed(yt,seed,budget=5)
        rocket_raw=MiniRocket(n_kernels=1008,random_state=seed,n_jobs=1)
        raw_train=rocket_raw.fit_transform(tr.transpose(0,2,1))
        raw_test=rocket_raw.transform(te.transpose(0,2,1))
        rocket_shape=MiniRocket(n_kernels=1008,random_state=seed,n_jobs=1)
        gauge_train=rocket_shape.fit_transform(x_train.transpose(0,2,1))
        gauge_test=rocket_shape.transform(x_test.transpose(0,2,1))
        raw_train,raw_test,rawsc=scale_pair(np.asarray(raw_train),np.asarray(raw_test))
        gauge_train,gauge_test,gsc=scale_pair(np.asarray(gauge_train),np.asarray(gauge_test))
        n_epochs=3 if args.smoke else args.epochs
        learned={}
        for mechanism in [False,True]:
            net,_=pretrain(x_train,seed,n_epochs,mechanism)
            learn_train,learn_test=encode(net,x_train),encode(net,x_test)
            la,lb,lsc=scale_pair(learn_train,learn_test)
            learned[mechanism]=(net,la,lb,lsc)
        models={}
        for name,mechanism in [('Raw-MiniRocket',None),('Gauge-MiniRocket',None),
                               ('Gauge+generic-SSL',False),('Gauge+mechanism-SSL',True)]:
            rocket_tr=raw_train if name=='Raw-MiniRocket' else gauge_train
            latent_tr=learned[mechanism][1] if mechanism is not None else np.zeros((len(tr),1))
            weights=[0.] if mechanism is None else [0.,.1,.3,1.,3.]
            models[name]=kernel_score(tr,yt,ids,{'latent':(latent_tr,None),'shape':(rocket_tr,None)},weights)
        basis=torch.tensor(harmonic_basis(x_test.shape[1]))
        for strength in [0.,.5,1.5]:
            shifted=generate_affine_shift(te,seed,strength)
            test_shape,concepts=affine_gauge(shifted)
            # Certified invariance only for additive affine perturbations, not general shifts.
            discrepancy=float(np.max(np.abs(test_shape-x_test)))
            assert discrepancy<2e-4
            raw_feat=rocket_raw.transform(shifted.transpose(0,2,1))
            rf=np.clip(rawsc.transform(np.asarray(raw_feat)),-10,10)/np.sqrt(raw_train.shape[1])
            shape_feat=rocket_shape.transform(test_shape.transpose(0,2,1))
            gf=np.clip(gsc.transform(np.asarray(shape_feat)),-10,10)/np.sqrt(gauge_train.shape[1])
            for name,mechanism in [('Raw-MiniRocket',None),('Gauge-MiniRocket',None),
                                   ('Gauge+generic-SSL',False),('Gauge+mechanism-SSL',True)]:
                if mechanism is None: latent=np.zeros((len(te),1))
                else:
                    model,_,__,sc=learned[mechanism]
                    latent=np.clip(sc.transform(encode(model,test_shape)),-10,10)/np.sqrt(encode(model,x_train).shape[1])
                prediction,margin=infer(models[name],latent,rf if name=='Raw-MiniRocket' else gf)
                row=dict(dataset=args.dataset,method=name,seed=seed,strength=strength,status='EXECUTED',
                         macro_f1=float(f1_score(yv,prediction,average='macro')),
                         accuracy=float(accuracy_score(yv,prediction)),
                         labelled_train=int(len(ids)),unlabelled_train=int(len(tr)),test_count=int(len(te)),
                         weight=models[name]['weight'],lambda_ridge=models[name]['ridge'],
                         train_cv=models[name]['cv_macro_f1'],gauge_max_err=discrepancy,
                         pretrained=False,epochs=n_epochs)
                allrows.append(row);print('RESULT_JSON='+json.dumps(row),flush=True)
        (output/'results.json').write_text(json.dumps(allrows,indent=2,allow_nan=False))
        print('SEED_DONE='+str(seed)+' elapsed_s='+str(round(time.time()-started,1)),flush=True)
    if any(r['status']!='EXECUTED' for r in allrows):raise RuntimeError('Missing score')
    print('TOTAL='+str(len(allrows)),flush=True)

if __name__=='__main__':main()
