from itertools import groupby
import numpy as np
import torch
from amortized_opt.rhythm import DrumKit
from isaac_rl.hand_plan_dataset import StochasticHandRhythmSampler,split_case_ids,dataset_metrics,extend_case_splits,load_dataset_batches
from isaac_rl.min_time_assignment import MinimumTimeChart


def sampler():
    kit=DrumKit.from_layout();allowed=np.ones((2,7,7),bool);allowed[0,6,:]=allowed[0,:,6]=False
    chart=MinimumTimeChart(tuple(kit.names),np.zeros((2,7,7)),np.zeros((2,7,7)),np.zeros((2,7,7)),allowed)
    return StochasticHandRhythmSampler(kit,4,chart)


def test_iid_sampling_supports_all_sets_mixed_intervals_and_single_plan_cases():
    s=sampler();clips=s.sample(4096,np.random.default_rng(98))
    groups=[];mixed=0;oneplan=0
    for clip in clips:
        rows=[list(g) for _,g in groupby(clip.hits,key=lambda h:h.time)]
        sets=[tuple(h.drum for h in g) for g in rows];groups.extend(sets)
        gaps=np.diff(np.r_[0.,[g[0].time for g in rows]])
        assert len(rows)==4 and (gaps>=.02-1e-7).all() and (gaps<=.90+1e-7).all()
        mixed+=gaps.min()<=.1 and gaps.max()>=.7
        oneplan+=np.prod([len(s.chart.choices(g)) for g in rows])==1
    assert len(set(groups))==28 and mixed>0 and oneplan>0
    assert .72<np.mean([len(g)==2 for g in groups])<.78
    assert clips==s.sample(4096,np.random.default_rng(98))


def test_case_splits_are_disjoint_complete_and_keep_whole_case_indices():
    splits=split_case_ids(8192)
    assert {k:len(v) for k,v in splits.items()}=={'train':6554,'validation':819,'test':819}
    allids=np.concatenate(list(splits.values()))
    np.testing.assert_array_equal(np.sort(allids),np.arange(8192))
    assert not set(splits['train'])&set(splits['test'])
    assert not set(splits['train'])&set(splits['validation'])


def test_training_extension_keeps_original_held_out_cases_exactly():
    old=split_case_ids(8192);new=extend_case_splits(old,8192,8192)
    np.testing.assert_array_equal(new['validation'],old['validation'])
    np.testing.assert_array_equal(new['test'],old['test'])
    assert len(new['train'])==14746
    assert not set(new['train'])&set(new['test'])
    assert not set(new['train'])&set(new['validation'])
    assert set(range(8192,16384))<=set(new['train'])


def test_manifest_keeps_old_case_indices_when_loading_more_data(tmp_path):
    import json
    sources=[]
    for name,offset in [('base',0),('extra',100)]:
        folder=tmp_path/name;(folder/'batches').mkdir(parents=True)
        obs=np.stack([np.full(136,offset+i,np.float32) for i in range(10)])
        outcomes={key:np.zeros((10,16),np.float32) for key in ('rewards','hit_rates','hits','requested')}
        torch.save(dict(batch={'observations':obs,'observed':np.ones((10,16),bool)},outcomes=outcomes),
                   folder/'batches/batch_00001.pt')
        sources.append(str(folder))
    combined=tmp_path/'combined';combined.mkdir()
    (combined/'manifest.json').write_text(json.dumps({'source_datasets':sources}))
    data=load_dataset_batches(combined)
    np.testing.assert_array_equal(data['observations'][:10,0],np.arange(10))
    np.testing.assert_array_equal(data['observations'][10:,0],np.arange(100,110))
    assert data['observed'].shape==(20,16)


def test_dataset_metrics_mask_unexecuted_plans_and_preserve_hit_totals():
    observed=np.zeros((2,16),bool);observed[:,:2]=True
    rewards=np.zeros((2,16),np.float32);rewards[0,:2]=[.2,.8];rewards[1,:2]=[-.3,-.1]
    hits=np.zeros((2,16),int);hits[0,:2]=[1,4]
    requested=np.zeros((2,16),int);requested[:,:2]=5
    scores=np.full((2,16),99.);scores[:,:2]=[[0,1],[0,1]]
    stats=dataset_metrics(scores,dict(observed=observed,rewards=rewards,hits=hits,requested=requested),np.arange(2))
    assert stats['hits']==4 and stats['requested']==10
    assert stats['hit_rate']==.4 and stats['mean_selected_actual_rank']==1
    assert stats['pair_accuracy']==1 and stats['mean_best_plan_predicted_rank']==1
