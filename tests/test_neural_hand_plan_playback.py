from types import SimpleNamespace
import numpy as np
import torch
from torch import nn

from amortized_opt.rhythm import DrumKit, Hit, RhythmClip, strike_speed
from isaac_rl.schedule import BatchedSchedule, ScheduleConfig
from isaac_rl.assignment_train import install_beats
from isaac_rl.neural_hand_plan_playback import NeuralHandPlanAssignment


class Scores(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(()))
        self.register_buffer("scores",torch.arange(16,dtype=torch.float32))

    def forward(self,obs):
        assert obs.shape == (1,136)
        return self.scores[None]+self.parameter*0


def fixture(mode="receding"):
    kit=DrumKit.from_layout()
    hits=[Hit(1.+i*.2,1.+i*.2,i%6,kit.names[i%6],"left",.6) for i in range(6)]
    hits.append(Hit(1.4,1.4,6,kit.names[6],"right",.8))
    clip=RhythmClip("six",hits,2.5)
    B,M=1,32
    u=SimpleNamespace(device="cpu",beat_valid=torch.zeros(B,M,dtype=torch.bool),
        beat_time=torch.zeros(B,M),beat_hand=torch.zeros(B,M,dtype=torch.long),
        beat_drum_id=torch.zeros(B,M,dtype=torch.long),beat_velocity=torch.zeros(B,M),
        beat_target_speed_m_s=torch.zeros(B,M),beat_pos=torch.zeros(B,M,3),
        beat_drum_height=torch.zeros(B,M),beat_dipole_offset=torch.zeros(B,M),
        beat_done=torch.zeros(B,M,dtype=torch.bool),
        drum_pos_env=torch.as_tensor(kit.positions,dtype=torch.float32)[None],
        drum_height=torch.as_tensor(kit.heights,dtype=torch.float32),
        drum_dipole_offset=torch.zeros(7),beat_is_left=torch.zeros(B,M,dtype=torch.bool),
        beat_is_right=torch.zeros(B,M,dtype=torch.bool),
        robot=SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(1,17),joint_vel=torch.ones(1,17)*.1)),
        scene=SimpleNamespace(env_origins=torch.tensor([[10.,20.,0.]])))
    u._compute_tip_state=lambda:(torch.tensor([[10.1,20.2,.8]]),torch.zeros(1,3),
                                torch.tensor([[10.1,19.8,.8]]),torch.zeros(1,3))
    agent=SimpleNamespace(u=u,kit=kit,_joint_ids=list(range(17)),
        schedule=BatchedSchedule(1,ScheduleConfig(num_drums=8,max_hits=32),device="cpu"),
        mpc=SimpleNamespace())
    model=Scores().eval()
    online=NeuralHandPlanAssignment(agent,model,clip,mode=mode)
    install_beats(u,online.queue)
    return agent,model,online


def test_live_state_four_group_selection_preserves_prefix_and_handles_tail():
    agent,model,online=fixture()
    online.advance(0.)
    assert len(online.records)==1
    assert online.records[0]["initial_drums"]==[-1,-1]
    np.testing.assert_allclose(online.records[0]["tips"],[[.1,.2,.8],[.1,-.2,.8]],atol=1e-6)
    first=online.queue.hands[0]
    assert first=="right"
    floor=next(i for i,h in enumerate(online.queue.hits) if h.drum==6)
    assert online.queue.hands[floor]=="right"
    model.scores.zero_();model.scores[0]=100
    online.advance(.5)
    assert len(online.records)==1  # No per-tick hand flicker.
    online.advance(1.01)
    assert online.queue.hands[0]==first and online.queue.hands[1]=="left"
    assert online.records[-1]["initial_drums"]==[-1,0]
    online.advance(1.21)
    assert set(online.queue.hands)==set(range(len(online.queue.hits)))
    count=len(online.records)
    online.advance(1.41);online.advance(1.61);online.advance(2.5)
    assert len(online.records)==count  # Last full window, no invented padding.
    assert online.queue.hands[floor]=="right"


def test_already_credited_future_group_cannot_change_hands():
    agent,model,online=fixture()
    online.advance(0.)
    old=online.queue.hands[1]
    agent.u.beat_done[0,1]=True
    model.scores.zero_();model.scores[0]=100
    online.advance(1.01)
    assert online.queue.hands[1]==old


def test_new_assignment_table_keeps_normalized_loudness_and_correct_speed_targets():
    agent,_,online=fixture()
    for i,h in enumerate(online.queue.hits):
        assert agent.u.beat_velocity[0,i].item()==np.float32(h.velocity)
        np.testing.assert_allclose(agent.u.beat_target_speed_m_s[0,i].item(),strike_speed(h.velocity),rtol=1e-6)


def test_reassigned_active_goal_invalidates_committed_key_without_erasing_reference():
    agent,_,online=fixture()
    agent.schedule.reset(torch.tensor([0]),[online.queue.clip],[agent.kit])
    agent.mpc._committed_goal_key=torch.tensor([[1,1]])
    reference=torch.ones(1,50,17)
    agent.mpc._committed_reference=reference
    online.advance(0.)
    assert torch.equal(agent.mpc._committed_goal_key,torch.tensor([[-1,-1]]))
    assert agent.mpc._committed_reference is reference


def test_four_group_block_is_immutable_until_last_deadline_and_tail_is_real_only():
    agent,model,online=fixture(mode="block")
    online.advance(0.)
    assert online.records[0]["committed_groups"]==[0,1,2,3]
    assert set(online.locked)=={0,1,2,3}
    assert len(online.queue.hands)==5  # Four groups, one simultaneous pair.
    original=dict(online.queue.hands)
    model.scores.zero_();model.scores[0]=100
    for time in [.5,1.01,1.21,1.41,1.59]:
        online.advance(time)
        assert online.queue.hands==original and len(online.records)==1
    online.advance(1.6)
    assert len(online.records)==2
    assert online.records[-1]["committed_groups"]==[4,5]
    assert online.records[-1]["padded_input_groups"]==2
    assert len(online.records[-1]["legal_actions"])==4  # No duplicate padded plans.
    assert set(online.queue.hands)==set(range(7))
    assert all(online.queue.hands[i]==hand for i,hand in original.items())
    online.advance(2.1)
    assert len(online.records)==2
