"""Small permutation-equivariant reranker over legal hand-plan candidates."""
import torch
from torch import nn

from isaac_rl.hand_plan_ranker import ACTION_BANK
from isaac_rl.hand_plan_dataset import dataset_metrics
from isaac_rl.hand_plan_value import SPLIT_NAMES, regression_metrics
import numpy as np


class CandidateAwarePlanReranker(nn.Module):
    """Frozen absolute-reward scorer plus a learned, mean-centered correction.

    Candidate self-attention is permutation equivariant. A separate correction
    head is fitted for each hit-scoring runtime; only the v4 head is used for
    hand assignment. The baseline remains unchanged at initialization.
    """
    candidate_aware = True

    def __init__(self, base, *, width=128, heads=4, layers=2, dropout=.1):
        super().__init__()
        if width < 1 or heads < 1 or width % heads or layers < 1:
            raise ValueError('Invalid candidate-attention dimensions')
        self.base = base.requires_grad_(False).eval()
        encoded_width=4*base.v4.config.group_width+base.v4.config.state_width
        self.input = nn.Sequential(nn.Linear(encoded_width+1+4,width),nn.LayerNorm(width),nn.GELU())
        layer = nn.TransformerEncoderLayer(d_model=width,nhead=heads,
            dim_feedforward=2*width,dropout=dropout,activation='gelu',batch_first=True)
        self.attention = nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)
        self.v4_correction = nn.Linear(width,1)
        self.v2_correction = nn.Linear(width,1)
        nn.init.zeros_(self.v4_correction.weight);nn.init.zeros_(self.v4_correction.bias)
        nn.init.zeros_(self.v2_correction.weight);nn.init.zeros_(self.v2_correction.bias)
        self.register_buffer('plans',torch.as_tensor(ACTION_BANK),persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, observations, *, runtime='v4', legal=None, plans=None,
                return_correction=False):
        if runtime not in ('v4','v2'):raise ValueError('Unknown reward runtime')
        if plans is None:plans=self.plans
        plans=torch.as_tensor(plans,device=observations.device,dtype=observations.dtype)
        if plans.ndim==2:plans=plans[None].expand(len(observations),-1,-1)
        if plans.ndim!=3 or plans.shape[0]!=len(observations) or plans.shape[2]!=4:
            raise ValueError('Expected [cases,candidates,4] assignment bits')
        shape=plans.shape[:2]
        if legal is None:legal=torch.ones(shape,device=observations.device,dtype=torch.bool)
        else:legal=torch.as_tensor(legal,device=observations.device,dtype=torch.bool)
        if legal.shape!=shape or not legal.any(1).all():
            raise ValueError('Each case needs at least one legal plan')
        with torch.no_grad():
            encoded=self.base.v4.candidate_scorer.encode(observations,plans)
            scorer=(self.base.v4.candidate_scorer.scorer if runtime=='v4'
                else self.base.v2_scorer)
            base_score=scorer(encoded).squeeze(-1)
        token=self.input(torch.cat((encoded,base_score[...,None],plans),dim=-1))
        hidden=self.attention(token,src_key_padding_mask=~legal)
        head=self.v4_correction if runtime=='v4' else self.v2_correction
        raw=head(hidden).squeeze(-1)
        count=legal.sum(1,keepdim=True)
        centered=(raw-(raw*legal).sum(1,keepdim=True)/count).masked_fill(~legal,0.)
        output=base_score+centered
        return (output,centered) if return_correction else output

    def score_observed_rows(self,observations,actions,legal,*,runtime='v4'):
        scores=self(observations,legal=legal,runtime=runtime)
        return scores.gather(1,torch.as_tensor(actions,device=scores.device)[:,None])[:,0]


@torch.inference_mode()
def evaluate_candidate_reranker(model,v4,v2,*,case_chunk=512,row_chunk=512):
    """Evaluate complete v4 rows and factual continuations with all legal plans present."""
    model.eval();device=next(model.parameters()).device
    cases=v4['cases'];points=v4['points']
    all_scores=np.zeros_like(cases['rewards'],dtype=np.float32)
    for start in range(0,len(cases['observations']),case_chunk):
        stop=min(start+case_chunk,len(cases['observations']))
        all_scores[start:stop]=model(torch.as_tensor(cases['observations'][start:stop],device=device),
            legal=torch.as_tensor(cases['observed'][start:stop],device=device)).cpu().numpy()
    primary={}
    for code,name in enumerate(SPLIT_NAMES):
        ids=np.asarray(v4['splits'][name],np.int64)
        selected=np.flatnonzero(points['splits']==code)
        factual=selected[points['source'][selected]==1]
        predictions=np.empty(len(selected),np.float32)
        base=points['source'][selected]==0
        predictions[base]=all_scores[points['case_ids'][selected[base]],points['actions'][selected[base]]]
        for start in range(0,len(factual),row_chunk):
            row_ids=factual[start:start+row_chunk]
            obs=torch.as_tensor(points['observations'][row_ids],device=device)
            legal=torch.as_tensor(cases['observed'][points['case_ids'][row_ids]],device=device)
            actions=torch.as_tensor(points['actions'][row_ids],device=device)
            values=model.score_observed_rows(obs,actions,legal).cpu().numpy()
            # selected is sorted, so factual positions can be found by search.
            predictions[np.searchsorted(selected,row_ids)]=values
        stats=regression_metrics(predictions,points['rewards'][selected])
        for source,label in ((0,'full'),(1,'continuation')):
            mask=points['source'][selected]==source
            values=regression_metrics(predictions[mask],points['rewards'][selected][mask])
            stats.update({f'{label}_{key}':value for key,value in values.items()})
        stats.update({f'selection_{key}':value for key,value in
            dataset_metrics(all_scores[ids],cases,ids).items()})
        primary[name]=stats
    old=v2['cases'];auxiliary={}
    for name in ('validation','test'):
        ids=np.asarray(v2['splits'][name],np.int64);scores=[]
        for start in range(0,len(ids),case_chunk):
            chunk=ids[start:start+case_chunk]
            scores.append(model(torch.as_tensor(old['observations'][chunk],device=device),runtime='v2',
                legal=torch.as_tensor(old['observed'][chunk],device=device)).cpu().numpy())
        scores=np.concatenate(scores)
        valid=old['observed'][ids]
        stats=regression_metrics(scores[valid],old['rewards'][ids][valid])
        stats.update({f'selection_{key}':value for key,value in dataset_metrics(scores,old,ids).items()})
        auxiliary[name]=stats
    return primary,auxiliary
