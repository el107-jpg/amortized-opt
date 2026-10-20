"""Soft hand-assignment priors on simulator-free rhythm/start collocations."""
from dataclasses import replace
import numpy as np
import torch

from amortized_opt.rhythm import Hit
from isaac_rl.assignment_search import hit_groups
from isaac_rl.hand_plan_ranker import ACTION_BANK, encode_clip, legal_actions


SNARE, HIHAT = 0, 1


def inject_snare_hihat(clip,kit,rng,mode):
    """Place the same two-drum motif into varied sampled windows."""
    if mode not in ('consecutive','simultaneous'):
        raise ValueError('Unknown snare/hi-hat motif mode')
    groups=hit_groups(clip)
    if len(groups)!=4:raise ValueError('Four hit groups required')
    if mode=='consecutive':
        first=int(rng.integers(3));drums=(SNARE,HIHAT)
        if rng.random()<.5:drums=drums[::-1]
        for offset,drum in enumerate(drums):
            original=groups[first+offset][0]
            groups[first+offset]=[Hit(original.time,original.raw_time,drum,
                kit.names[drum],'left',float(rng.uniform(.2,1.)))]
    else:
        slot=int(rng.integers(4));original=groups[slot][0]
        groups[slot]=[Hit(original.time,original.raw_time,drum,kit.names[drum],
            ('left','right')[index],float(rng.uniform(.2,1.)))
            for index,drum in enumerate((SNARE,HIHAT))]
    return replace(clip,name=clip.name+'_'+mode+'_sh',hits=[hit for group in groups for hit in group])


def heuristic_plan_values(clip,initial_drums,*,motif_mode='both',
                          motif_margin=.06,switch_margin=.006):
    """Reward-unit margins for canonical snare/hi-hat and hand drum reuse."""
    if motif_mode not in ('both','consecutive','simultaneous'):
        raise ValueError('Unknown motif matching mode')
    if len(initial_drums)!=2:raise ValueError('Two initial hand targets required')
    groups=hit_groups(clip)
    if len(groups)!=4:raise ValueError('Four hit groups required')
    assigned=np.full((16,4,2),-1,np.int8)
    rows=np.arange(16)
    for group_index,group in enumerate(groups):
        bits=ACTION_BANK[:,group_index]
        assigned[rows,group_index,bits]=group[0].drum
        if len(group)==2:assigned[rows,group_index,1-bits]=group[1].drum
    motif_errors=np.zeros(16,np.float32);motifs=0
    if motif_mode in ('both','simultaneous'):
        for index,group in enumerate(groups):
            if len(group)==2 and {hit.drum for hit in group}=={SNARE,HIHAT}:
                motif_errors+=(assigned[:,index,0]!=SNARE).astype(np.float32)
                motif_errors+=(assigned[:,index,1]!=HIHAT).astype(np.float32)
                motifs+=1
    if motif_mode in ('both','consecutive'):
        for index in range(3):
            left,right=groups[index:index+2]
            if len(left)==len(right)==1 and {left[0].drum,right[0].drum}=={SNARE,HIHAT}:
                snare_index=index if left[0].drum==SNARE else index+1
                hihat_index=index if left[0].drum==HIHAT else index+1
                motif_errors+=(assigned[:,snare_index,0]!=SNARE).astype(np.float32)
                motif_errors+=(assigned[:,hihat_index,1]!=HIHAT).astype(np.float32)
                motifs+=1
    switches=np.zeros(16,np.float32)
    previous=np.tile(np.asarray(initial_drums,np.int8),(16,1))
    for index in range(4):
        for hand in range(2):
            drum=assigned[:,index,hand]
            present=drum>=0
            switches+=((previous[:,hand]>=0)&present&(previous[:,hand]!=drum))
            previous[present,hand]=drum[present]
    return -(motif_margin*motif_errors+switch_margin*switches),dict(motifs=motifs,
        mean_switches=float(switches.mean()))


def sample_collocations(sampler,kit,eligibility,training_start_features,rng,count,
                        *,motif_mode='both',motif_probability=.5,
                        motif_margin=.06,switch_margin=.006):
    """Fresh IID rhythms paired with independently sampled train-only starts."""
    if count<1 or not 0<=motif_probability<=1:raise ValueError('Invalid collocation sampling')
    starts=np.asarray(training_start_features,np.float32)
    if starts.ndim!=2 or starts.shape[1]!=56 or len(starts)<1:
        raise ValueError('Expected train-only 56-value initial-state features')
    clips=sampler.sample(count,rng,minimum_plan_choices=1)
    start_ids=rng.integers(len(starts),size=count)
    observations=[];legal=[];prior=[];motif_count=0;switches=[]
    for index,clip in enumerate(clips):
        if rng.random()<motif_probability:
            insertion=motif_mode if motif_mode!='both' else ('consecutive','simultaneous')[int(rng.integers(2))]
            clip=inject_snare_hihat(clip,kit,rng,insertion)
        features=encode_clip(clip,tips=np.zeros((2,3),np.float32))
        features[80:]=starts[start_ids[index]]
        allowed=legal_actions(clip,eligibility)
        initial=(int(starts[start_ids[index],34:42].argmax())-1,
                 int(starts[start_ids[index],42:50].argmax())-1)
        values,stats=heuristic_plan_values(clip,initial,motif_mode=motif_mode,
            motif_margin=motif_margin,switch_margin=switch_margin)
        observations.append(features);legal.append(allowed);prior.append(values)
        motif_count+=stats['motifs'];switches.append(stats['mean_switches'])
    return (np.stack(observations),np.stack(legal),np.stack(prior),
        dict(cases=count,motifs=motif_count,mean_switches=float(np.mean(switches))))


def collocation_prior_loss(predicted,prior,legal,*,scale=.05):
    """Squared one-sided violations of heuristic reward-gap margins."""
    if predicted.shape!=prior.shape or legal.shape!=predicted.shape:
        raise ValueError('Collocation score/heuristic/mask shape differs')
    if predicted.ndim!=2 or scale<=0 or not legal.any(1).all():
        raise ValueError('Invalid collocation prior batch')
    legal=legal.bool()
    desired=prior[:,:,None]-prior[:,None,:]
    actual=predicted[:,:,None]-predicted[:,None,:]
    active=(desired>1e-7)&legal[:,:,None]&legal[:,None,:]
    violation=(desired-actual).clamp_min(0)
    per_case=(violation.square()*active).sum((1,2))/active.sum((1,2)).clamp_min(1)
    return per_case.mean()/scale,dict(active_pairs=int(active.sum().detach()),
        violated_pairs=int((active&(violation>0)).sum().detach()),
        mean_margin=float(desired[active].mean().detach()) if active.any() else 0.)
