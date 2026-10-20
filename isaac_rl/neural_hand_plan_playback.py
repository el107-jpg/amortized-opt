"""State-conditioned four-group neural hand selection for live playback."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from isaac_rl.arm_assignment import AssignmentConfig, AssignmentQueue
from isaac_rl.hand_plan_ranker import ACTION_BANK, HandPlanRanker, HandPlanRankerConfig, encode_clip


def load_hand_plan_policy(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = HandPlanRankerConfig(**checkpoint["network"])
    if cfg.architecture != "shared_candidate" or cfg.scorer_depth != 2:
        raise ValueError("This playback requires the requested two-layer shared-candidate scorer")
    if checkpoint["format"] == "mixed_runtime_hand_plan_value_v1":
        from isaac_rl.hand_plan_value import DualRuntimeRewardModel
        model = DualRuntimeRewardModel(cfg).to(device)
    elif checkpoint["format"] == "offline_hand_plan_value_v1":
        model = HandPlanRanker(cfg).to(device)
    else:
        raise ValueError("Unsupported neural hand-plan checkpoint format")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)
    return model, dict(format=checkpoint["format"], step=checkpoint.get("step",checkpoint.get("gradient_steps")),
                       network=checkpoint["network"], inference_runtime="v4")


class NeuralHandPlanAssignment:
    """Commit fixed four-group blocks, or explicitly use receding selection.

    Block mode scores once and holds all four hands through the last deadline.
    A final partial block pads only the input and executes only real hits.
    Receding mode is retained explicitly for reproducing the prior experiment.
    """
    def __init__(self, agent, model, clip, *, trace_path=None, mode="block"):
        if mode not in {"block","receding"}:
            raise ValueError("Unknown hand-plan commitment mode")
        self.mode = mode
        self.agent, self.model = agent, model
        self.queue = AssignmentQueue(clip, agent.kit, AssignmentConfig())
        if not self.queue.groups or (mode=="receding" and len(self.queue.groups)<4):
            raise ValueError("Playback needs hit groups; receding mode needs at least four")
        self.floor = agent.kit.index("floor_tom")
        self.locked = {}
        self.last_window = None
        self.next_block = 0
        self.block_end_time = -float("inf")
        self.records = []
        self.trace_path = Path(trace_path) if trace_path else None

    def initial_drums(self, now):
        previous = [-1, -1]
        for index, hit in enumerate(self.queue.hits):
            if hit.time > now + 1e-8:
                break
            hand = self.queue.hands.get(index)
            if hand is not None:
                previous[int(hand == "right")] = hit.drum
        return previous

    def legal_mask(self, start):
        legal = np.ones(16, bool)
        for offset in range(4):
            group = start + offset
            if group >= len(self.queue.groups):
                # Padding is input-only, never an executed hit. Force unused
                # bits to zero so partial tails have unique legal candidates.
                legal &= ACTION_BANK[:,offset] == 0
                continue
            ids = self.queue.groups[group]
            for slot, index in enumerate(ids):
                if self.queue.hits[index].drum == self.floor:
                    legal &= (ACTION_BANK[:,offset] + slot) % 2 == 1
            if group in self.locked:
                legal &= ACTION_BANK[:,offset] == self.locked[group]
        if not legal.any():
            raise ValueError("No mechanically legal neural hand plan")
        return legal

    def _read_state(self):
        u = self.agent.u
        q = u.robot.data.joint_pos[0,self.agent._joint_ids].float().detach().cpu().numpy()
        qd = u.robot.data.joint_vel[0,self.agent._joint_ids].float().detach().cpu().numpy()
        left, _, right, _ = u._compute_tip_state()
        tips = (torch.stack((left[0],right[0]))-u.scene.env_origins[0]).detach().cpu().numpy()
        return q, qd, tips

    def _protect_credited(self):
        done = getattr(self.agent.u,"beat_done",None)
        if done is None:
            return
        values = done[0].detach().cpu().numpy()
        for group, ids in enumerate(self.queue.groups):
            if any(values[i] for i in ids) and ids[0] in self.queue.hands:
                self.locked[group] = int(self.queue.hands[ids[0]] == "right")

    @torch.inference_mode()
    def advance(self, now):
        from isaac_rl.assignment_train import sync_planner
        self._protect_credited()
        if self.mode=="block":
            if now+1e-8 < self.block_end_time or self.next_block >= len(self.queue.groups):
                return
            start = self.next_block
        else:
            future = [g for g,ids in enumerate(self.queue.groups)
                      if self.queue.hits[ids[0]].time > now+1e-8]
            if not future:
                return
            start = future[0]
            if start==self.last_window or start > len(self.queue.groups)-4:
                return
        count = min(4,len(self.queue.groups)-start)
        q, qd, tips = self._read_state()
        indices = [i for group in self.queue.groups[start:start+4] for i in group]
        window = replace(self.queue.clip, hits=[replace(self.queue.hits[i],
            time=self.queue.hits[i].time-now) for i in indices])
        previous = self.initial_drums(now)
        obs = encode_clip(window,q=q,qd=qd,initial_drums=previous,tips=tips,
                          allow_partial=self.mode=="block")
        device = next(self.model.parameters()).device
        scores = self.model(torch.as_tensor(obs[None],device=device)).float().cpu().numpy()[0]
        if scores.shape != (16,) or not np.isfinite(scores).all():
            raise FloatingPointError("Invalid neural hand-plan scores")
        legal = self.legal_mask(start)
        action = int(np.argmax(np.where(legal,scores,-np.inf)))
        bits = ACTION_BANK[action].astype(int)
        for offset, bit in enumerate(bits[:count]):
            group = start+offset
            for slot,index in enumerate(self.queue.groups[group]):
                hand = ("left","right")[(int(bit)+slot)%2]
                if self.queue.hits[index].drum == self.floor and hand != "right":
                    raise AssertionError("LH floor assignment escaped legal mask")
                self.queue.hands[index] = hand
        if self.mode=="block":
            self.locked.update({start+i:int(bit) for i,bit in enumerate(bits[:count])})
            self.next_block = start+count
            self.block_end_time = self.queue.hits[self.queue.groups[self.next_block-1][0]].time
        else:
            self.locked[start] = int(bits[0])
        if self.mode=="receding" and start==len(self.queue.groups)-4:
            self.locked.update({start+i:int(bit) for i,bit in enumerate(bits)})
        self.last_window = start
        self.records.append(dict(time_s=float(now),first_group=start,action=action,
            bits=bits[:count].tolist(),committed_groups=list(range(start,start+count)) if self.mode=="block" else [start],
            padded_input_groups=4-count,legal_actions=np.flatnonzero(legal).tolist(),scores=scores.tolist(),
            predicted_reward=float(scores[action]),initial_drums=previous,
            observations=obs.tolist(),q=q.tolist(),qd=qd.tolist(),tips=tips.tolist()))
        before = [value.detach().clone() for value in self.agent.schedule.goal()]
        sync_planner(self.agent,self.queue)
        if any(not torch.equal(old,new) for old,new in zip(before,self.agent.schedule.goal())):
            # A reassigned active goal may keep its old cursor index. Its old
            # committed trajectory must not survive a drum/deadline/hand change.
            key = getattr(self.agent.mpc,"_committed_goal_key",None)
            if key is not None:
                self.agent.mpc._committed_goal_key = torch.full_like(key,-1)

    def finish(self):
        if self.trace_path:
            payload=dict(method="two_layer_neural_four_group_"+self.mode,inference_runtime="v4",
                decisions=self.records,locked_groups=self.locked,
                hits=[dict(time=h.time,drum=h.drum,hand=self.queue.hands.get(i))
                      for i,h in enumerate(self.queue.hits)])
            self.trace_path.parent.mkdir(parents=True,exist_ok=True)
            self.trace_path.write_text(json.dumps(payload,indent=2)+"\n")
