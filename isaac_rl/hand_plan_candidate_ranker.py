"""Shared drum/hit encoders and explicit candidate-conditioned plan scoring."""
from itertools import product
import torch
from torch import nn


class SharedCandidateScorer(nn.Module):
    """Map the existing136-value observation and binary plans to scalar scores.

    Each slot uses the same drum embedding/hit encoder; each group uses the
    same encoder after its hits are arranged as left/right by the candidate.
    The final scorer sees the ordered four-group sequence and full initial
    state, retaining interactions between hands, waist and different groups.
    """
    def __init__(self,hidden=224,dropout=0.,group_width=64,state_width=64,scorer_depth=2):
        super().__init__()
        if min(group_width,state_width)<1:raise ValueError('Encoder widths must be positive')
        if scorer_depth not in (2,3):raise ValueError('Scorer supports two or three hidden layers')
        self.drum_embedding=nn.Embedding(7,16)
        self.hit_encoder=nn.Sequential(nn.Linear(18,32),nn.GELU(),nn.Dropout(dropout))
        self.group_encoder=nn.Sequential(nn.Linear(67,group_width),nn.LayerNorm(group_width),nn.GELU(),nn.Dropout(dropout))
        self.state_encoder=nn.Sequential(nn.Linear(56,state_width),nn.LayerNorm(state_width),nn.GELU(),nn.Dropout(dropout))
        scorer=[nn.Linear(4*group_width+state_width,hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Dropout(dropout)]
        if scorer_depth==3:scorer.extend((nn.Linear(hidden,hidden),nn.GELU(),nn.Dropout(dropout)))
        scorer.extend((nn.Linear(hidden,max(1,hidden//2)),nn.GELU(),nn.Dropout(dropout),
            nn.Linear(max(1,hidden//2),1)))
        self.scorer=nn.Sequential(*scorer)
        nn.init.normal_(self.scorer[-1].weight,std=.001);nn.init.zeros_(self.scorer[-1].bias)
        self.register_buffer('action_bank',torch.tensor(list(product((0.,1.),repeat=4))),persistent=False)

    def encode(self,observations,plans=None):
        if observations.ndim!=2 or observations.shape[1]!=136:raise ValueError('Expected [cases,136] observations')
        batch=len(observations)
        if plans is None:plans=self.action_bank
        else:
            plans=torch.as_tensor(plans,device=observations.device,dtype=observations.dtype)
            if not ((plans==0)|(plans==1)).all():raise ValueError('Plans must contain binary assignments')
        if plans.ndim==2 and plans.shape[1]==4:plans=plans[None].expand(batch,-1,-1)
        if plans.ndim!=3 or plans.shape[0]!=batch or plans.shape[2]!=4:
            raise ValueError('Expected [candidates,4] or [cases,candidates,4] plans')
        count=plans.shape[1]
        groups=observations[:,:80].reshape(batch,4,20)
        slots=groups[:,:,:18].reshape(batch,4,2,9)
        embedding=slots[...,:7]@self.drum_embedding.weight
        hits=self.hit_encoder(torch.cat((embedding,slots[...,7:]),dim=-1))*slots[...,8:9]
        first,second=hits[:,None,:,0,:],hits[:,None,:,1,:]
        bits=plans[...,None]
        left=first*(1-bits)+second*bits;right=first*bits+second*(1-bits)
        times=groups[:,None,:,18:].expand(-1,count,-1,-1)
        encoded=self.group_encoder(torch.cat((left,right,times,bits),dim=-1))
        state=self.state_encoder(observations[:,80:])[:,None].expand(-1,count,-1)
        return torch.cat((encoded.flatten(2),state),dim=-1)

    def forward(self,observations,plans=None):
        return self.scorer(self.encode(observations,plans)).squeeze(-1)
