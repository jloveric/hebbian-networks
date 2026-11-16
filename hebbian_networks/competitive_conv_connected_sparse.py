"""Sparse accumulation in original order with unchanged connected coactivity and checkpoints."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from dataclasses import asdict
from functools import lru_cache

import numpy as np
from . import competitive_conv_connected_groups as base
from . import common

REPO=base.REPO
ROOT=common.BUILD/'cifar_sparse'
SOURCE=common.KERNELS/'competitive_conv_connected_sparse.mojo'
save_checkpoint=base.save_checkpoint
load_checkpoint=base.load_checkpoint
Config=base.Config
data=base.data
fingerprint=base.fingerprint
initialize=base.initialize
forward=base.forward
geometry=base.geometry
score_weights=base.score_weights
selected_scores=base.selected_scores
mask_plan=base.mask_plan
training_scales=base.training_scales


def source_paths():
    return sorted(set(base.source_paths()) | set(common.source_paths(SOURCE, [Path(__file__)])))


@lru_cache(None)
def backend():
    return common.backend("competitive_conv_connected_sparse_native", SOURCE, source_paths())


def _native(state,images,c,labels=None,trace=False,reverse=False,masks=None,evaluation_batch=None,splits=None):
    x=np.ascontiguousarray(images,np.float32)
    if x.ndim!=4 or not len(x) or x.shape[1]!=x.shape[2] or not np.isfinite(x).all():raise ValueError('invalid images')
    sides=geometry(c,x.shape[1])
    if len(state)!=len(c.channels) or (trace and len(x)>c.batch):raise ValueError('invalid state or trace size')
    cin=x.shape[-1]
    for bank,width,kernel in zip(state,c.channels,c.kernels):
        shape=(10,kernel*kernel*cin+1,width)
        if any(bank[key].shape!=shape or bank[key].dtype!=np.float32 or not bank[key].flags.c_contiguous or not np.isfinite(bank[key]).all() for key in ('w','m')):
            raise ValueError('invalid connected weight/history geometry')
        if bank['clock'].shape!=(10,width) or bank['clock'].dtype!=np.int64 or (bank['clock']<0).any():raise ValueError('invalid clocks')
        cin=10*width
    y=np.zeros(len(x),np.int64) if labels is None else np.ascontiguousarray(labels,np.int64)
    if y.shape!=(len(x),) or ((y<0)|(y>=10)).any():raise ValueError('invalid labels')
    if evaluation_batch is not None and (labels is not None or not 1<=evaluation_batch<=128):raise ValueError('evaluation batch is inference only')
    if labels is not None and c.dropout_probability:
        if masks is None or len(masks)!=len(state):raise ValueError('explicit deterministic masks required')
        for i in range(1,len(state)):
            a=masks[i]
            if a.shape!=(len(x),10*c.channels[i-1]) or not np.isfinite(a).all():raise ValueError('invalid shared feature masks')
    splits=[16 if i==0 else 4 if s*s>=16 else 1 for i,s in enumerate(sides)] if splits is None else list(splits)
    if len(splits)!=len(state) or any(type(s) is not int or s<1 for s in splits):raise ValueError('invalid accumulation splits')
    packet=dict(state=state,images=x,labels=y,train=labels is not None,trace=trace,reverse=reverse,
        config=asdict(c),splits=splits,evaluation_batch=c.batch if evaluation_batch is None else evaluation_batch,
        energies=np.zeros((len(x),10,len(state)),np.float32),health=np.ones(1,np.float32),
        utilization=[np.zeros((10,w),np.float32) for w in c.channels],
        positive_utilization=[np.zeros((10,w),np.float32) for w in c.channels],
        scales=np.zeros((len(x),10),np.float32),score_weights=score_weights(c,x.shape[1]),
        masks=[np.ones(1,np.float32) if i==0 or masks is None or labels is None else np.ascontiguousarray(masks[i],np.float32) for i in range(len(state))])
    if trace:
        packet['records']=[dict(u=np.zeros((1,len(x),s,s,b['w'].shape[1]),np.float32),
              h=np.zeros((10,len(x),s,s,w),np.float32),orientation=np.zeros((10,len(x),s,s,w),np.float32))
              for b,s,w in zip(state,sides,c.channels)]
        if labels is not None:packet['pooled_statistics']=[dict(numerator=np.zeros_like(b['w']),nq=np.zeros((10,w,2),np.float32)) for b,w in zip(state,c.channels)]
    native,_=backend();native.run(packet)
    if packet['health'][0]!=1 or not np.isfinite(packet['energies']).all():raise FloatingPointError('nonfinite connected convolution state/activity')
    return packet


def score(state,images,c=Config(),trace=False,evaluation_batch=None):
    return _native(state,images,c,trace=trace,evaluation_batch=evaluation_batch)


def train(state,images,labels,c=Config(),trace=False,reverse=False,masks=None,splits=None):
    return _native(state,images,c,labels,trace,reverse,masks,splits=splits)

