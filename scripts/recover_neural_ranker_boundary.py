#!/usr/bin/env python3
"""Replay saved labels/RNG to audit a historical ranker; never update live weights."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from amortized_opt.rhythm import DrumKit
from isaac_rl.hand_plan_ranker import HandPlanRanker,HandPlanRankerConfig,RankerReplay,ranking_loss,encode_clip
from isaac_rl.min_time_assignment import MinimumTimeChart
from isaac_rl.chart_cma_sampling import ProceduralChartRhythmSampler
from scripts.train_neural_hand_plans import make_start_bank,learning_rate,load_cached_panel,evaluate_cached


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--update',type=int,default=10)
    p.add_argument('--out',type=Path,required=True)
    cli=p.parse_args()
    torch.set_num_threads(2)
    cfg=json.loads((cli.run/'config.json').read_text());args=cfg['args']
    saved=torch.load(cli.run/'initial.pt',map_location='cpu',weights_only=False)
    model=HandPlanRanker(HandPlanRankerConfig(**saved['network']));model.load_state_dict(saved['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=args['lr'],weight_decay=1e-4)
    optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['torch_rng'])
    rng=np.random.default_rng();rng.bit_generator.state=saved['rng']
    bank=make_start_bank(model.config.tip_geometry)
    if saved.get('post_hit_bank'):bank.load_state_dict(saved['post_hit_bank'])
    # Bootstrap generated two uniform velocities per trial on each eligible
    # hand/drum; the simulator RNG lives in separate processes.
    eligible=MinimumTimeChart.load(cfg['eligibility_source'])
    for file in sorted(cli.run.glob('bootstrap_*.json')):
        rng.uniform(.3,.9,size=(13*args['bootstrap_repeats']*2))
        record=json.loads(file.read_text())
        metrics=bank.add([s for r in record['reports'] for s in r.get('post_hit_states',[])],rng)
        for key in ('accepted','bank_size','combinations','counts'):
            if metrics.get(key)!=record['metrics'].get(key):raise RuntimeError(f'Bootstrap replay differs: {key}')
    replay=RankerReplay(args['replay_capacity'])
    sampler=ProceduralChartRhythmSampler(DrumKit.from_layout(),4,eligible)
    audit=[];step=0
    for update in range(1,cli.update+1):
        record=torch.load(cli.run/f'trials_{update:05d}.pt',map_location='cpu',weights_only=False)
        batch,outcomes=record['batch'],record['outcomes']
        panel=sampler.sample(args['contexts'],rng,minimum_plan_choices=2)
        if [c.name for c in panel]!=[c.name for c in batch['panel']]:raise RuntimeError('Rhythm RNG replay differs')
        states,_=bank.sample(rng,args['contexts'])
        for actual,wanted in zip(states,batch['states']):
            for key in ('q','qd','tips'):
                np.testing.assert_array_equal(actual[key],wanted[key])
        model.eval()
        with torch.inference_mode():scores=model(torch.as_tensor(batch['observations'])).numpy()
        error=float(np.abs(scores-batch['scores']).max())
        if error>1e-6:raise RuntimeError(f'Historical scores differ at update{update}: {error}')
        reports=json.loads((cli.run/f'episodes_{update:05d}.json').read_text())
        bank.add([s for r in reports for s in r.get('post_hit_states',[])],rng)
        replay.add(batch['observations'],outcomes,batch['observed'])
        model.train()
        for _ in range(args['fit_steps']):
            step+=1
            lr=learning_rate(step,sum(args['stages'])*args['fit_steps'],args['lr'],args['min_lr'],args['warmup'])
            for group in optimizer.param_groups:group['lr']=lr
            draw=replay.sample(rng,args['batch'])
            loss,_=ranking_loss(model(draw['observations']),draw['rewards'],draw['observed'])
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        audit.append(dict(update=update,pre_fit_saved_score_max_error=error))
    # Verify the next collection's saved pre-fit scores too: this is the model
    # actually used when switching, not just an approximate shadow refit.
    following=cli.run/f'trials_{cli.update+1:05d}.pt'
    model.eval()
    if following.exists():
        batch=torch.load(following,map_location='cpu',weights_only=False)['batch']
        with torch.inference_mode():scores=model(torch.as_tensor(batch['observations'])).numpy()
        error=float(np.abs(scores-batch['scores']).max())
        if error>1e-6:raise RuntimeError(f'Boundary model fails saved next-batch score check: {error}')
    else:error=None
    held=load_cached_panel(Path(args['cached_panel']),cfg['checkpoint_sha256'])
    held['observations']=np.stack([encode_clip(c,tip_geometry=model.config.tip_geometry) for c in held['panel']])
    with torch.inference_mode():scores=model(torch.as_tensor(held['observations'])).numpy()
    valid,rewards=held['observed'],held['rewards']
    chosen=np.where(valid,scores,-np.inf).argmax(1)
    selected_rewards=rewards[np.arange(len(chosen)),chosen]
    ranks=1+((rewards>selected_rewards[:,None]+1e-8)&valid).sum(1)
    full=ranks[valid.sum(1)==16]
    metrics=evaluate_cached(model,held)
    original=json.loads((cli.run/f'iteration_{cli.update:05d}.json').read_text())['held_out']
    for key in ('total_hits','mean_selected_actual_reward_rank','mean_best_reward_predicted_rank'):
        np.testing.assert_allclose(metrics[key],original[key],rtol=0,atol=1e-8)
    result=dict(boundary_update=cli.update,next_batch_score_max_error=error,audit=audit,
        held_out_mean_rank=float(ranks.mean()),held_out_population_std_rank=float(ranks.std()),
        held_out_sample_std_rank=float(ranks.std(ddof=1)),held_out_contexts=len(ranks),
        full16_contexts=len(full),full16_mean_rank=float(full.mean()),full16_std_rank=float(full.std()),
        rank1_fraction=float((ranks==1).mean()),held_out_metrics=metrics,
        rank_histogram={str(k):int((ranks==k).sum()) for k in np.unique(ranks)},
        note='Recovered from exact stored outcomes and RNG; verified saved next-batch scores and original held-out metrics. Live model unchanged.')
    cli.out.mkdir(parents=True,exist_ok=True)
    (cli.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(cli.out/'ranks.npz',ranks=ranks,legal_counts=valid.sum(1),selected_actions=chosen)
    torch.save(dict(network=asdict(model.config),model=model.state_dict(),update=cli.update,audit_only=True),cli.out/'recovered_model.pt')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
