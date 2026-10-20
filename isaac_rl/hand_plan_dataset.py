"""IID rhythm cases and whole-case supervised dataset splits."""
from itertools import combinations
import numpy as np
import torch
from amortized_opt.rhythm import Hit,RhythmClip
from isaac_rl.chart_cma_sampling import ProceduralChartRhythmSampler


class StochasticHandRhythmSampler:
    """Independent choices of drum set, every interval and strike velocity.

    Uniform over the28 possible nonempty single/two-distinct-drum sets; each
    interval is independently uniform over20–900ms in20ms ticks. No forced
    transition templates, rhythm-wide tempo class or coverage stratification.
    """
    def __init__(self,kit,groups,eligibility):
        self.kit,self.groups,self.chart=kit,int(groups),eligibility
        if tuple(kit.names)!=eligibility.names or groups!=4:
            raise ValueError('IID16-action collection requires four groups on this kit')
        self.sets=[tuple(c) for size in (1,2) for c in combinations(range(kit.num_drums),size)]
        self.last_strata=None
        self.summary=dict(source='iid_stochastic_four_group_windows',groups=groups,
            drum_set_distribution='uniform_over_7_singles_and_21_distinct_pairs',
            intervals_s=[.02,.90],grid_s=.02,velocities=[.2,1.],
            first_hit_policy='iid_random',forced_request_anchors=False,balanced_timing_tiers=False)

    def sample(self,count,rng,minimum_plan_choices=1):
        if minimum_plan_choices not in (1,2):raise ValueError('Unsupported legal-plan minimum')
        clips,tiers=[],[]
        for _ in range(count):
            # A one-plan case is valid (e.g. four floor requests); collection
            # defaults to1, so random sampling does not discard these rhythms.
            while True:
                gaps=rng.integers(1,46,size=self.groups)
                times=np.cumsum(gaps)*.02;hits=[];choice_count=1
                for t in times:
                    drums=self.sets[int(rng.integers(len(self.sets)))]
                    rows=[Hit(round(float(t),8),round(float(t),8),d,self.kit.names[d],
                        ('left','right')[slot],float(rng.uniform(.2,1.))) for slot,d in enumerate(drums)]
                    choices=self.chart.choices(rows)
                    if not choices:raise ValueError('IID drum set has no mechanical assignment')
                    choice_count*=len(choices);hits.extend(rows)
                if choice_count>=minimum_plan_choices:break
            clips.append(RhythmClip(f'iid_hand_{int(rng.integers(2**63)):016x}',hits,hits[-1].time+.3))
            median=float(np.median(gaps)*.02)
            tiers.append(0 if median<=.12 else 1 if median<=.32 else 2)
        self.last_strata=np.asarray(tiers)
        return clips

    def seed_payload(self):
        return ProceduralChartRhythmSampler(self.kit,self.groups,self.chart).seed_payload()


def split_case_ids(count,seed=202610121):
    if count<10:raise ValueError('At least10 whole cases required for dataset split')
    ids=np.random.default_rng(seed).permutation(count)
    nval=max(1,int(round(count*.1)));ntest=nval
    return dict(train=ids[:count-nval-ntest],validation=ids[count-nval-ntest:count-ntest],test=ids[count-ntest:])


def load_dataset_batches(folder):
    from pathlib import Path
    import json
    manifest=Path(folder)/'manifest.json'
    if manifest.exists():
        sources=json.loads(manifest.read_text())['source_datasets']
        if not sources:raise ValueError('Combined dataset has no sources')
        parts=[load_dataset_batches(Path(source)) for source in sources]
        return {key:np.concatenate([part[key] for part in parts]) for key in parts[0]}
    files=sorted((Path(folder)/'batches').glob('batch_*.pt'))
    if not files:raise ValueError('Dataset contains no completed batches')
    data={k:[] for k in ('observations','rewards','hit_rates','hits','requested','observed')}
    for file in files:
        saved=torch.load(file,map_location='cpu',weights_only=False);batch,out=saved['batch'],saved['outcomes']
        data['observations'].append(batch['observations']);data['observed'].append(batch['observed'])
        for key in ('rewards','hit_rates','hits','requested'):data[key].append(out[key])
    return {key:np.concatenate(values) for key,values in data.items()}


def load_prefirst_factual_rows(folder):
    """Load v4 single-plan continuation rows without crossing case splits."""
    from pathlib import Path
    folder=Path(folder)
    files=sorted((folder/'batches').glob('batch_*.pt'))
    if not files:raise ValueError('No v4 factual batches found')
    keys=('observations','actions','rewards','hit_rates','case_ids','elapsed_s','splits')
    columns={key:[] for key in keys}
    split_ids=torch.load(folder/'splits.pt',map_location='cpu',weights_only=False)
    code_by_case=np.full(sum(len(v) for v in split_ids.values()),-1,np.int8)
    for code,name in enumerate(('train','validation','test')):
        code_by_case[np.asarray(split_ids[name],np.int64)]=code
    if (code_by_case<0).any():raise ValueError('Whole-case split does not cover dataset')
    for path in files:
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if 'factual' not in saved:raise ValueError(f'{path} has no factual rows')
        data=saved['factual'];batch=saved['batch']
        case_ids=np.asarray(data['case_ids'],np.int64)
        local=case_ids-int(batch['case_offset'])
        if len(local) and ((local<0).any() or (local>=len(batch['panel'])).any()):
            raise ValueError('Factual row refers outside its source case batch')
        if len(local) and not batch['observed'][local,np.asarray(data['actions'],np.int64)].all():
            raise ValueError('Factual row uses an unobserved hand assignment')
        if len(local) and not np.array_equal(data['splits'],code_by_case[case_ids]):
            raise ValueError('Factual row crossed a whole-case split')
        for key in keys:columns[key].append(np.asarray(data[key]))
    result={key:np.concatenate(values) for key,values in columns.items()}
    if (result['observations'].ndim!=2 or result['observations'].shape[1]!=136
            or not np.isfinite(result['observations']).all()
            or not np.isfinite(result['rewards']).all()
            or (result['observations'][:,19]<=0).any()):
        raise ValueError('Invalid pre-first-hit factual observation or label')
    return result


def extend_case_splits(original,original_count,new_count):
    old={key:np.asarray(ids,dtype=np.int64) for key,ids in original.items()}
    if not np.array_equal(np.sort(np.concatenate(list(old.values()))),np.arange(original_count)):
        raise ValueError('Original splits do not partition all cases')
    return dict(train=np.r_[old['train'],np.arange(original_count,original_count+new_count)],
                validation=old['validation'].copy(),test=old['test'].copy())


def rollout_runtime_fingerprint(config):
    """Older cached labels used the contact-reset bug and live scoring defaults."""
    runtime=dict(rollout_runtime_version='legacy_contact_reset_v0',fixed_hit_scoring=None,decoder_amp='torch.bfloat16',impact_speed_target=None)
    runtime.update(config.get('rollout_runtime') or {})
    return runtime


def require_matching_rollout_runtime(first,second):
    if rollout_runtime_fingerprint(first)!=rollout_runtime_fingerprint(second):
        raise ValueError('Dataset rollout runtime differs; regenerate labels instead of mixing legacy and corrected contact resets/scoring')


def dataset_metrics(scores,data,ids):
    valid=data['observed'][ids];rewards=data['rewards'][ids];scores=np.asarray(scores)
    chosen=np.where(valid,scores,-np.inf).argmax(1);rows=np.arange(len(ids))
    best=np.where(valid,rewards,-np.inf).max(1);selected=rewards[rows,chosen]
    ranks=1+((rewards>selected[:,None]+1e-8)&valid).sum(1)
    diffs=rewards[:,:,None]-rewards[:,None,:];pred=scores[:,:,None]-scores[:,None,:]
    mask=valid[:,:,None]&valid[:,None,:]&np.triu(np.ones((16,16),bool),1)[None]&(np.abs(diffs)>.05)
    ties=np.abs(pred)<1e-8;correct=((pred*diffs)>0).astype(float);correct[ties]=.5
    hits=data['hits'][ids][rows,chosen];requested=data['requested'][ids][rows,chosen]
    order=np.argsort(-np.where(valid,scores,-np.inf),axis=1,kind='stable')
    ordered=np.take_along_axis(rewards,order,1);best_rank=np.argmax(ordered>=best[:,None]-1e-8,axis=1)+1
    return dict(cases=len(ids),hits=int(hits.sum()),requested=int(requested.sum()),
        hit_rate=float(hits.sum()/requested.sum()),mean_reward=float(selected.mean()),
        mean_reward_regret=float((best-selected).mean()),mean_selected_actual_rank=float(ranks.mean()),
        mean_best_plan_predicted_rank=float(best_rank.mean()),best_plan_fraction=float((selected>=best-1e-8).mean()),
        pair_accuracy=float(correct[mask].sum()/max(1,mask.sum())),strong_pairs=int(mask.sum()))
