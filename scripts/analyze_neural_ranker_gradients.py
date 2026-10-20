#!/usr/bin/env python3
"""Separate weighted best-set and pairwise gradient contributions without fitting."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from isaac_rl.hand_plan_ranker import HandPlanRanker,HandPlanRankerConfig


def losses(scores,rewards,observed):
    best=rewards.masked_fill(~observed,-torch.inf).max(1,keepdim=True).values
    winners=observed & (rewards >= best-.02)
    q=winners.float()/winners.sum(1,keepdim=True)
    logs=F.log_softmax(scores.masked_fill(~observed,-1e9),1)
    best_loss=2.*(-(q*logs).sum(1)).mean()
    pairs=observed[:,:,None]&observed[:,None,:]
    pairs &= torch.ones(16,16,dtype=torch.bool).triu(1)[None]
    gaps=rewards[:,:,None]-rewards[:,None,:]
    target=torch.sigmoid(gaps/.05)
    weight=(gaps.abs()/.1).clamp(.05,1.)*pairs
    ce=F.binary_cross_entropy_with_logits(scores[:,:,None]-scores[:,None,:],target,reduction='none')
    pair_loss=((ce*weight).sum((1,2))/weight.sum((1,2)).clamp_min(1e-8)).mean()
    return best_loss,pair_loss


def vector_stats(a,b):
    best=float(a.norm());pair=float(b.norm());total=a+b
    dot=float(a@b);denom=float(total@total)
    return dict(best_norm=best,ordering_norm=pair,
        ordering_norm_share=pair/max(best+pair,1e-30),
        best_vs_ordering_cosine=dot/max(best*pair,1e-30),
        ordering_projection_on_total=(float(b@total)/denom if denom>1e-30 else None),
        best_projection_on_total=(float(a@total)/denom if denom>1e-30 else None),
        total_norm=float(total.norm()))


def measure(model,data,ids):
    observations=torch.as_tensor(data['observations'][ids])
    rewards=torch.as_tensor(data['rewards'][ids]);valid=torch.as_tensor(data['observed'][ids])
    scores=model(observations);best,pair=losses(scores,rewards,valid)
    params=list(model.parameters())
    best_param=torch.cat([g.reshape(-1) for g in torch.autograd.grad(best,params,retain_graph=True)])
    pair_param=torch.cat([g.reshape(-1) for g in torch.autograd.grad(pair,params,retain_graph=True)])
    best_score=torch.autograd.grad(best,scores,retain_graph=True)[0].reshape(-1)
    pair_score=torch.autograd.grad(pair,scores)[0].reshape(-1)
    return dict(weighted_best_loss=float(best.detach()),ordering_loss=float(pair.detach()),
        parameter_gradients=vector_stats(best_param,pair_param),
        score_gradients=vector_stats(best_score,pair_score))


def summarize_batches(rows):
    out={}
    for space in ('parameter_gradients','score_gradients'):
        out[space]={}
        for key in rows[0][space]:
            values=np.array([r[space][key] for r in rows],float)
            out[space][key]=dict(mean=float(values.mean()),std=float(values.std()),
                median=float(np.median(values)),p10=float(np.quantile(values,.1)),p90=float(np.quantile(values,.9)))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--batches',type=int,default=64);p.add_argument('--seed',type=int,default=93844)
    cli=p.parse_args();torch.set_num_threads(2)
    checkpoint=torch.load(cli.run/'last.pt',map_location='cpu',weights_only=False)
    model=HandPlanRanker(HandPlanRankerConfig(**checkpoint['network']));model.load_state_dict(checkpoint['model']);model.eval()
    data=checkpoint['replay'].data;rng=np.random.default_rng(cli.seed)
    records=[measure(model,data,rng.integers(len(data['observations']),size=128)) for _ in range(cli.batches)]
    summary=dict(update=checkpoint['completed_updates'],minibatches=cli.batches,minibatch_size=128,
        replay_rhythms=len(data['observations']),loss='2*best-set CE + normalized weighted pairwise BCE',
        full_replay=measure(model,data,np.arange(len(data['observations']))),
        minibatch_statistics=summarize_batches(records),
        norm_share_note='Ordering norm/(ordering norm+weighted best norm); vectors overlap so this is not a literal fraction of the AdamW update.',
        projection_note='Signed projection on combined raw gradient sums to1; negative means opposition. Before clipping, AdamW moments and weight decay.')
    # Additional check at the16-plan boundary, using its replay at that time.
    recovered=cli.run/'top8_boundary_audit/recovered_model.pt'
    if recovered.exists():
        past=torch.load(recovered,map_location='cpu',weights_only=False)
        old=HandPlanRanker(HandPlanRankerConfig(**past['network']));old.load_state_dict(past['model']);old.eval()
        pieces=[torch.load(cli.run/f'trials_{i:05d}.pt',map_location='cpu',weights_only=False) for i in range(1,11)]
        earlier=dict(observations=np.concatenate([x['batch']['observations'] for x in pieces]),
            rewards=np.concatenate([x['outcomes']['rewards'] for x in pieces]),
            observed=np.concatenate([x['batch']['observed'] for x in pieces]))
        observations=[measure(old,earlier,rng.integers(len(earlier['observations']),size=128)) for _ in range(cli.batches)]
        summary['update10_minibatch_statistics']=summarize_batches(observations)
    cli.out.mkdir(parents=True,exist_ok=True)
    (cli.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (cli.out/'minibatches.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
