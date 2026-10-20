"""Controlled local perturbations and finite-grid sensitivity statistics."""
from dataclasses import replace
import copy
import numpy as np
from isaac_rl.assignment_search import hit_groups
from isaac_rl.hand_plan_ranker import cartesian_tips

AXES=('first_delay','second_gap','waist_yaw','arm_direction')


def perturb_case(clip,state,axis,offset,arm_direction,geometry):
    """offset is seconds for timing axes and radians for pose axes; no clipping."""
    if axis not in AXES:raise ValueError('Unknown sweep axis')
    changed=copy.deepcopy(state)
    if axis in AXES[:2]:
        groups=hit_groups(clip);times=np.array([g[0].time for g in groups])
        times[(0 if axis=='first_delay' else 1):]+=offset
        if (np.diff(np.r_[0.,times])<=1e-9).any():return None,None,'Nonpositive first delay or inter-group gap'
        hits=[replace(h,time=round(float(t),9),raw_time=round(float(h.raw_time+offset*(axis=='first_delay' or i>=1)),9))
              for i,(g,t) in enumerate(zip(groups,times)) for h in g]
        return replace(clip,hits=hits,duration=float(clip.duration+offset)),changed,None
    q=np.asarray(state['q'],np.float32).copy()
    if axis=='waist_yaw':q[0]+=offset
    else:q+=np.asarray(arm_direction,np.float32)*offset
    changed['q']=q.tolist();changed['tips']=cartesian_tips(q,geometry).tolist()
    return clip,changed,None


def sensitivity_summary(points,reports,*,fresh_scene_baselines=False):
    """Neighbor differences are finite-grid evidence, not continuity proofs."""
    baseline={};axis_rows={axis:[] for axis in AXES}
    separate_baseline_slots=any('control_axis' in p for p in points)
    for point in points:
        if not point['valid']:continue
        rows=[reports[i] for i in point['report_ids']]
        reward=np.array([r['reward'] for r in rows]);hits=np.array([r['hits'] for r in rows])
        hit_masks=np.array([[h['hit'] for h in r['per_hit_outcomes']] for r in rows],bool)
        point['reward']=reward.tolist();point['hits']=hits.tolist()
        point['near_miss_bonus']=[r['near_miss_bonus'] for r in rows]
        point['physical_best_action']=point['actions'][int(reward.argmax())]
        point['best_reward']=float(reward.max())
        point['neural_selected_reward']=float(reward[point['actions'].index(point['neural_action'])])
        point['neural_reward_regret']=point['best_reward']-point['neural_selected_reward']
        baseline_eligible=not (fresh_scene_baselines and point.get('control_kind')=='control' and point.get('repeat',0)>0)
        if baseline_eligible and (point['axis']=='control' or point['step']==0):
            key=(point['case'],point.get('control_axis',point['axis'])) if separate_baseline_slots else point['case']
            baseline.setdefault(key,[]).append(reward)
        if point['axis'] in AXES:axis_rows[point['axis']].append((point,reward,hits,hit_masks))
    ranges=[np.ptp(np.stack(rows),axis=0) for rows in baseline.values()]
    noise=np.concatenate(ranges) if ranges else np.zeros(1)
    noise_bound=float(noise.max())
    result=dict(repeatability=dict(baseline_case_plan_pairs=len(noise),
        reward_range_median=float(np.median(noise)),reward_range_p95=float(np.quantile(noise,.95)),
        reward_range_max=noise_bound),axes={})
    for axis,rows in axis_rows.items():
        differences=[];count_changes=[];credit_changes=[];strict_flips=material_flips=neural_flips=neighbors=0
        reversals=triplets=0
        for case in sorted({p['case'] for p,*_ in rows}):
            ordered=sorted([r for r in rows if r[0]['case']==case],key=lambda r:r[0]['step'])
            for a,b in zip(ordered,ordered[1:]):
                if b[0]['step']-a[0]['step']!=1:continue
                neighbors+=1;differences.extend(np.abs(b[1]-a[1]))
                count_changes.extend(a[2]!=b[2]);credit_changes.extend((a[3]!=b[3]).any(1))
                strict_flips+=a[0]['physical_best_action']!=b[0]['physical_best_action']
                material_flips+=not np.any((a[1]>=a[1].max()-.02)&(b[1]>=b[1].max()-.02))
                neural_flips+=a[0]['neural_action']!=b[0]['neural_action']
            for a,b,c in zip(ordered,ordered[1:],ordered[2:]):
                if b[0]['step']-a[0]['step']!=1 or c[0]['step']-b[0]['step']!=1:continue
                reversals+=int(((a[3]==c[3]).all(1)&(a[3]!=b[3]).any(1)).sum());triplets+=len(a[1])
        delta=np.array(differences);n=len(delta)
        result['axes'][axis]=dict(case_point_neighbors=neighbors,plan_neighbors=n,
            adjacent_step_ms=1 if axis in AXES[:2] else None,adjacent_step_deg=.1 if axis in AXES[2:] else None,
            reward_difference_median=float(np.median(delta)) if n else None,
            reward_difference_p95=float(np.quantile(delta,.95)) if n else None,
            reward_difference_max=float(delta.max()) if n else None,
            fraction_reward_change_over_005=float(np.mean(delta>.05)) if n else None,
            fraction_reward_change_above_repeatability_bound=float(np.mean(delta>noise_bound+1e-6)) if n else None,
            fraction_hit_count_changed=float(np.mean(count_changes)) if n else None,
            fraction_hit_credit_changed=float(np.mean(credit_changes)) if n else None,
            strict_best_action_switches=int(strict_flips),disjoint_near_best_set_switches=int(material_flips),
            neural_action_switches=int(neural_flips))
        result['axes'][axis].update(one_point_hit_credit_reversals=reversals,plan_triplets=triplets,
            fraction_one_point_hit_credit_reversals=reversals/max(1,triplets))
    return result
