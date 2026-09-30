"""One-off health check for the feasible-QP terminal-pose ablation."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amortized_opt.terminal_pose import (
    TerminalPoseAL,
    TerminalPoseConfig,
    TerminalPoseCost,
    TerminalPoseNet,
    sample_terminal_batch,
)


RUNS = {
    "naked": (
        "runs/terminal_pose/ablate_feasible_qpth_20260926/qpout_10k.pt",
        None,
        True,
    ),
    "mix32": (
        "runs/terminal_pose/ablate_feasible_qpth_20260926/mix32_10k.pt",
        32,
        False,
    ),
    "mix128": (
        "runs/terminal_pose/ablate_feasible_qpth_20260926/mix128_10k.pt",
        128,
        False,
    ),
}
HEADS = ("to_chol", "to_p", "to_G", "to_l", "to_u", "to_A", "to_b")


def module_norm(module, attr="data"):
    terms = []
    for parameter in module.parameters():
        value = getattr(parameter, attr)
        if value is not None:
            terms.append(value.pow(2).sum())
    return float(sum(terms, torch.zeros(())).sqrt())


def main():
    batch = sample_terminal_batch(
        64,
        rng=np.random.default_rng(20260926),
        device="cpu",
        dtype=torch.float32,
    )
    result = {}
    for name, (path, mix_width, qp_out) in RUNS.items():
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        torch.manual_seed(0)
        initial = TerminalPoseNet(
            optnet=True,
            n_constraints=4,
            mix_width=mix_width,
            qp_out=qp_out,
        )
        torch.manual_seed(0)
        net = TerminalPoseNet(
            optnet=True,
            n_constraints=4,
            mix_width=mix_width,
            qp_out=qp_out,
        )
        net.load_state_dict(ckpt["model"])

        h1 = F.silu(net.pre(batch.features))
        h2 = F.silu(net.enc(h1))
        Q, _p, A, b, G, lower, upper, z0 = net.qp.qp_data(h2)
        z = net.qp(h2, net.q_lo, net.q_hi, net.q_ready)
        gz = (G @ z.unsqueeze(-1)).squeeze(-1)
        eq_res = (A @ z.unsqueeze(-1)).squeeze(-1) - b
        ineq_viol = torch.maximum(lower - gz, gz - upper).clamp_min(0)
        min_ineq_slack = torch.minimum(gz - lower, upper - gz)
        eig = torch.linalg.eigvalsh(Q)
        a_sv = torch.linalg.svdvals(A)
        g_sv = torch.linalg.svdvals(G)

        delta = {}
        activation_std = {}
        for head in HEADS:
            trained_head = getattr(net.qp, head)
            initial_head = getattr(initial.qp, head)
            delta[head] = float(
                sum(
                    (trained - untrained).pow(2).sum()
                    for trained, untrained in zip(
                        trained_head.parameters(), initial_head.parameters()
                    )
                ).sqrt()
            )
            activation_std[head] = float(trained_head(h2).std())

        al = TerminalPoseAL()
        al.load_state_dict(ckpt["al"])
        cost = TerminalPoseCost(
            TerminalPoseConfig(), device="cpu", dtype=torch.float32
        )
        loss = cost(net(batch.features), batch, al.cost_weights()).total.mean()
        net.zero_grad(set_to_none=True)
        loss.backward()
        grad = {
            head: module_norm(getattr(net.qp, head), "grad") for head in HEADS
        }

        residual = None
        if mix_width is not None:
            mixed = F.silu(
                net.mix(torch.cat([z.detach().float(), h2[:, : net.skip_dim]], dim=-1))
            )
            raw = net.head(mixed)
            r = net.q_half * torch.tanh(raw)
            residual = {
                "mean_abs_rad": float(r.abs().mean()),
                "mean_half_range_fraction": float((r.abs() / net.q_half).mean()),
                "max_half_range_fraction": float((r.abs() / net.q_half).max()),
            }

        center = (G @ z0.unsqueeze(-1)).squeeze(-1)
        result[name] = {
            "qp_param_delta_norm": delta,
            "qp_head_activation_std": activation_std,
            "qp_head_grad_norm": grad,
            "q_eigen_min": float(eig.min()),
            "q_condition_median": float((eig[:, -1] / eig[:, 0]).median()),
            "a_singular_min": float(a_sv.min()),
            "a_condition_median": float((a_sv[:, 0] / a_sv[:, -1]).median()),
            "g_singular_min": float(g_sv.min()),
            "g_condition_median": float((g_sv[:, 0] / g_sv[:, -1]).median()),
            "active_inequality_rows_mean": float(
                (min_ineq_slack < 1e-4).float().sum(dim=-1).mean()
            ),
            "active_inequality_fraction": float((min_ineq_slack < 1e-4).float().mean()),
            "equality_residual_max": float(eq_res.abs().max()),
            "inequality_violation_max": float(ineq_viol.max()),
            "witness_margin_min": float(
                torch.minimum(center - lower, upper - center).min()
            ),
            "mixer_residual": residual,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
