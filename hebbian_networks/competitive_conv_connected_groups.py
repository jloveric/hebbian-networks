"""One connected convolutional network with class-owned receiving filters."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from . import data
from .common import fingerprint
from .conv_reference import pool
from .conv_reference import reference_update as explicit_update

REPO=Path(__file__).resolve().parents[1]
VERSION='competitive-conv-connected-class-groups-1'


@dataclass(frozen=True)
class Config:
    channels: tuple=(80,80,80)  # Per class group; deeper filters see ten times this.
    winners: tuple=(5,5,5)
    kernels: tuple=(3,3,5)
    pool_after: tuple=(True,True,False)
    columns: int=10
    batch: int=32
    rate: float=.0005
    margin: float=.05
    dropout_probability: float=0.
    direction_history: float=.125
    history_decay: float=.99
    observation_balance: str='image'
    integration: str='semi_implicit'

    def __post_init__(self):
        for name in ('channels','winners','kernels','pool_after'):
            object.__setattr__(self,name,tuple(getattr(self,name)))
        if (self.integration not in ('semi_implicit','explicit')
                or self.columns!=10 or not 1<=len(self.channels)<=4
                or not len(self.channels)==len(self.winners)==len(self.kernels)==len(self.pool_after)
                or any(type(w) is not int or type(k) is not int or not 1<=k<=w<=256 for w,k in zip(self.channels,self.winners))
                or any(type(k) is not int or k<1 or k%2!=1 for k in self.kernels)
                or any(type(p) is not bool for p in self.pool_after) or self.pool_after[-1]
                or type(self.batch) is not int or not 1<=self.batch<=128
                or not np.isfinite(self.rate) or self.rate<=0
                or not np.isfinite(self.margin) or not 0<=self.margin<=1
                or not np.isfinite(self.dropout_probability) or not 0<=self.dropout_probability<1
                or self.observation_balance!='image'
                or any(not np.isfinite(v) or not 0<=v<1 for v in (self.direction_history,self.history_decay))):
            raise ValueError('invalid connected convolution configuration')


def geometry(c,side=32):
    result=[]
    for kernel,pooled in zip(c.kernels,c.pool_after):
        side=side-kernel+1
        if side<1 or (pooled and side<2):raise ValueError('image too small for convolutions/pooling')
        result.append(side)
        if pooled:side//=2
    return tuple(result)


def score_weights(c,side=32):
    return np.array([1/(s*s*k) for s,k in zip(geometry(c,side),c.winners)],np.float32)


def selected_scores(energy,c,side=32):
    return np.sum(energy*score_weights(c,side)[None,None,:],axis=-1,dtype=np.float32)


def initialize(c=Config(),seed=10101,input_channels=3):
    rngs=[np.random.default_rng(np.random.SeedSequence([seed,k])) for k in range(10)]
    state=[]
    for width,kernel in zip(c.channels,c.kernels):
        dimension=kernel*kernel*input_channels
        weights=[]
        for rng in rngs:
            w=rng.normal(size=(dimension,width))
            w/=np.linalg.norm(w,axis=0,keepdims=True)
            weights.append(np.concatenate((w,np.zeros((1,width))),axis=0).astype(np.float32))
        w=np.stack(weights)
        state.append(dict(w=w,m=np.zeros_like(w),clock=np.zeros((10,width),np.int64)))
        input_channels=10*width
    return state


def mask_plan(c,seed,epoch,count):
    return [None]+[(np.random.default_rng(np.random.SeedSequence([seed,epoch,l])).random((count,10*c.channels[l-1]))>=c.dropout_probability).astype(np.float32)/np.float32(1-c.dropout_probability)
                  for l in range(1,len(c.channels))]


def reflected_indices(dimension,kernel):
    channels=(dimension-1)//(kernel*kernel)
    return np.concatenate((np.arange(dimension-1).reshape(kernel,kernel,channels)[:,::-1].reshape(-1),[dimension-1]))


def forward(state,images,c=Config(),masks=None):
    """Independent label-blind patch extraction and NumPy projections."""
    features=np.asarray(images,np.float32)
    geometry(c,features.shape[1])
    records=[];energies=[]
    for layer,(bank,kernel,width,k) in enumerate(zip(state,c.kernels,c.channels,c.winners)):
        batch,side,_,cin=features.shape
        out_side=side-kernel+1
        u=np.ones((batch,out_side,out_side,kernel*kernel*cin+1),np.float32)
        for y in range(out_side):
            for x in range(out_side):
                u[:,y,x,:-1]=features[:,y:y+kernel,x:x+kernel].reshape(batch,-1)
        if layer==0:
            u=(u/np.sqrt(np.sum(u.astype(np.float64)**2,axis=-1,keepdims=True))).astype(np.float32)
        elif masks is not None:
            u[...,:-1]*=np.tile(masks[layer],(1,kernel*kernel))[:,None,None,:]
        flip=reflected_indices(u.shape[-1],kernel)
        fields=[];orientations=[]
        for group in range(10):
            ordinary=u@bank['w'][group]
            mirror=u@bank['w'][group,flip]
            use=(np.abs(mirror)>np.abs(ordinary))|((np.abs(mirror)==np.abs(ordinary))&(mirror>ordinary))
            fields.append(np.where(use,mirror,ordinary))
            orientations.append(use.astype(np.float32))
        z=np.stack(fields)
        selected=np.argsort(-np.abs(z),axis=-1,kind='stable')[...,:k]
        h=np.zeros_like(z)
        np.put_along_axis(h,selected,np.take_along_axis(z,selected,-1),-1)
        records.append(dict(u=u[None],h=h,orientation=np.stack(orientations)))
        energies.append(np.sum(h.astype(np.float64)**2,axis=(2,3,4)).T.astype(np.float32))
        transmitted=pool(h,'antialias') if c.pool_after[layer] else h
        # Every next-layer patch includes every preceding class group's channels.
        features=transmitted.transpose(1,2,3,0,4).reshape(batch,transmitted.shape[2],transmitted.shape[3],10*width)
    return records,np.stack(energies,-1)


def training_scales(scores,labels,c):
    rival_scores=scores.copy();rival_scores[np.arange(len(labels)),labels]=-np.inf
    rival=rival_scores.argmax(1)
    good=scores[np.arange(len(labels)),labels];bad=rival_scores.max(1)
    included=(scores.argmax(1)!=labels)|((good-bad)/(good+bad+np.float32(1e-30))<np.float32(c.margin))
    scales=np.zeros_like(scores)
    ids=np.flatnonzero(included)
    scales[ids,labels[ids]]=1;scales[ids,rival[ids]]=-1
    return scales


def reference_train(state,images,labels,c,reverse=False,masks=None):
    records,energy=forward(state,images,c,masks)
    scales=training_scales(selected_scores(energy,c,images.shape[1]),labels,c)
    pending=explicit_update(state,records,labels,c,reverse=reverse,scales=scales)
    return records,energy,scales,pending


def source_paths():
    from . import competitive_conv_connected_sparse as sparse
    return sparse.source_paths()


def backend():
    from . import competitive_conv_connected_sparse as sparse
    return sparse.backend()


def score(state, images, c=Config(), trace=False, evaluation_batch=None):
    from . import competitive_conv_connected_sparse as sparse
    return sparse.score(state, images, c, trace=trace, evaluation_batch=evaluation_batch)


def train(state, images, labels, c=Config(), trace=False, reverse=False, masks=None, splits=None):
    from . import competitive_conv_connected_sparse as sparse
    return sparse.train(state, images, labels, c, trace=trace, reverse=reverse, masks=masks, splits=splits)


def save_checkpoint(path,state,c,mean,progress):
    metadata=dict(version=VERSION,config=asdict(c),progress=progress,state_sha256=fingerprint(state))
    arrays={f'layer{i}_{key}':a for i,bank in enumerate(state) for key,a in bank.items()}
    data.atomic_npz(path,metadata=json.dumps(metadata),mean=mean,**arrays)


def load_checkpoint(path):
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata'].item()))
        if meta['version']!=VERSION:raise ValueError('unknown connected convolution checkpoint')
        c=Config(**meta['config'])
        state=[{key:z[f'layer{i}_{key}'].copy() for key in ('w','m','clock')} for i in range(len(c.channels))]
        mean=z['mean'].copy()
    if fingerprint(state)!=meta['state_sha256']:raise ValueError('checkpoint state mismatch')
    return state,c,mean,meta['progress']
