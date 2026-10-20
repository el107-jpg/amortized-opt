import torch

from amortized_opt.constants import ready_joint_pos
from amortized_opt.fk import ChainFK
from isaac_rl.schedule_distillation import (FourHitScheduleScorer, schedule_bank,
    suffix_table, value_and_ranking_loss, relative_score_loss)


def table():
    return dict(drums=torch.tensor([[[0,-1],[1,-1],[2,-1],[3,-1]]]),
        times=torch.tensor([[.16,.3,.5,.7]]), valid=torch.ones(1,4,dtype=torch.bool),
        speeds=torch.tensor([[[2.,0.]]*4]))


def test_features_use_measured_tip_velocity_and_ignore_physical_slot_orientation():
    torch.set_num_threads(1)
    model = FourHitScheduleScorer(); fk = ChainFK()
    q = torch.tensor(ready_joint_pos()[None], dtype=torch.float32)
    v = torch.linspace(-.2,.3,17)[None]
    t = table(); x = model.features(t,q,v,fk)
    assert x.shape == (1,86) and torch.isfinite(x).all()
    flipped = dict(t, drums=t['drums'].flip(-1), speeds=t['speeds'].flip(-1))
    torch.testing.assert_close(x,model.features(flipped,q,v,fk))
    eps = 1e-3
    after = model.features(t,q+eps*v,torch.zeros_like(v),fk)
    before = model.features(t,q-eps*v,torch.zeros_like(v),fk)
    actual = x[:,40:46]*3
    estimated = (after[:,34:40]-before[:,34:40])/(2*eps)
    torch.testing.assert_close(actual,estimated,atol=8e-5,rtol=2e-3)
    hits=x[:,46:].reshape(1,4,10)
    torch.testing.assert_close(hits[0,:,7],t['times'][0])
    torch.testing.assert_close(hits[0,:,8],torch.full((4,),2/3))


def test_suffix_time_action_and_padding_correspond_to_remaining_schedule():
    model = FourHitScheduleScorer(); fk = ChainFK()
    t = table(); q = torch.tensor(ready_joint_pos()[None], dtype=torch.float32)
    for g in range(1,5):
        suffix = suffix_table(t,t['times'][:,g-1],g)
        assert int(suffix['valid'].sum()) == 4-g
        assert (suffix['times'][~suffix['valid']] == 0).all()
        if g<4:
            torch.testing.assert_close(suffix['times'][0,:4-g], t['times'][0,g:]-t['times'][0,g-1])
        for c in range(16):
            index = (c % (1 << (4-g))) << g
            expected = torch.cat((schedule_bank()[c,g:],torch.zeros(g,dtype=torch.long)))
            torch.testing.assert_close(schedule_bank()[index],expected)
        x = model.features(suffix,q,torch.zeros_like(q),fk)
        _,actions=model.choose(x)
        assert (actions[:,4-g:] == 0).all()
        if g==4:
            torch.testing.assert_close(model(x),torch.zeros(1,16))


def test_joint_value_ranking_ignores_exact_ties_and_trains_all_outputs():
    target=torch.arange(16,dtype=torch.float32)[None].repeat(2,1)/20
    predicted=torch.zeros_like(target,requires_grad=True)
    value,ranking=value_and_ranking_loss(predicted,target)
    (value+.1*ranking).backward()
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad[0,-1]<predicted.grad[0,0]
    _,tie_ranking=value_and_ranking_loss(predicted.detach(),torch.ones_like(target))
    assert tie_ranking==0


def test_relative_loss_ignores_window_baselines_but_penalizes_wrong_differences():
    target = torch.arange(32, dtype=torch.float32).reshape(2, 16)/20
    offsets = torch.tensor([[10.], [-7.]])
    shifted = (target+offsets).requires_grad_()
    torch.testing.assert_close(relative_score_loss(shifted, target), torch.tensor(0.), atol=1e-10, rtol=0)
    prediction = target.flip(-1).clone().requires_grad_()
    loss = relative_score_loss(prediction, target)
    torch.testing.assert_close(loss, relative_score_loss(prediction+offsets, target-offsets))
    loss.backward()
    assert loss > 0 and prediction.grad[0, 0] > 0 and prediction.grad[0, -1] < 0
    torch.testing.assert_close(prediction.grad.sum(-1), torch.zeros(2), atol=1e-7, rtol=0)


def test_focused_ranking_compares_teacher_best_and_weights_large_gaps():
    target = torch.arange(16, dtype=torch.float32)[None]/2
    prediction = torch.linspace(1., -1., 16)[None].requires_grad_()
    _, ranking = value_and_ranking_loss(prediction, target, top_k=1, gap_cap=4.)
    gaps = (target[:, -1:]-target[:, :-1]).clamp_max(4.)
    penalties = torch.nn.functional.softplus((prediction[:, :-1]-prediction[:, -1:])/.5)
    expected = (gaps*penalties).sum()/gaps.sum()
    torch.testing.assert_close(ranking, expected)
    ranking.backward()
    assert prediction.grad[0, -1] < 0
    assert prediction.grad[0, 0] > prediction.grad[0, -2] > 0


def test_focused_ranking_handles_ties_and_averages_windows_equally():
    prediction = torch.zeros(2, 16, requires_grad=True)
    target = torch.stack((torch.arange(16, dtype=torch.float32)/20, torch.ones(16)))
    _, combined = value_and_ranking_loss(prediction, target, top_k=4)
    _, live = value_and_ranking_loss(prediction[:1], target[:1], top_k=4)
    torch.testing.assert_close(combined, live/2)
    combined.backward()
    assert torch.isfinite(prediction.grad).all() and (prediction.grad[1] == 0).all()


def test_configurable_width_checkpoints_and_legacy_loader(tmp_path):
    for widths in ((32,48,24),(256,256,128)):
        model = FourHitScheduleScorer(hidden_dims=widths).eval()
        cp = dict(format=model.format,model=model.state_dict(),reward_scale=model.reward_scale)
        if widths!=(256,256,128):
            cp['hidden_dims']=widths
        path=tmp_path/'scorer.pt';torch.save(cp,path)
        restored=FourHitScheduleScorer.from_checkpoint(path)
        assert restored.hidden_dims==widths
        x=torch.randn(2,86);x[:,[55,65,75,85]]=1.
        torch.testing.assert_close(restored(x),model(x))
        torch.testing.assert_close(restored.choose(x)[1],model.choose(x)[1])


def test_mixed_shards_combine_only_available_suffix_transitions(tmp_path):
    from scripts.train_four_hit_schedule import load_split
    for i in range(2):
        row=dict(features=torch.zeros(2,86),values=torch.zeros(2,16),graph_index=torch.zeros(2,dtype=torch.long),
            root_ids=[f'{i}:0',f'{i}:1'])
        if i==1:
            row['transitions']=dict(obs=torch.ones(3,86))
        torch.save(row,tmp_path/f'{i}.pt')
    data=load_split(tmp_path,torch.device('cpu'))
    assert data['features'].shape==(4,86) and len(data['root_ids'])==4
    torch.testing.assert_close(data['transitions']['obs'],torch.ones(3,86))
