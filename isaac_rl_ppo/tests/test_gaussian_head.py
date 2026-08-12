"""The Gaussian head: exact log-probs, AR(1) marginals, healthy KL/entropy.

These are the tests that decide whether PPO is *valid* here at all.  If the
AR(1) sampler's marginal drifts from ``N(mean, std)``, the stored log-prob is
not the density the action was drawn from, the importance ratio is biased, and
every subsequent number in the run is meaningless while looking completely
normal.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.distributions import Normal

from isaac_rl_ppo.actor import build_ar1_rho


# =============================================================== log-prob ====
def test_log_prob_matches_stock_normal(actor):
    """The head's log-prob IS ``Normal(mean, sigma).log_prob``, summed."""
    torch.manual_seed(0)
    mean = 0.3 * torch.randn(64, 17)
    dist = actor.update_distribution(mean)
    actions = dist.sample()

    got = actor.get_actions_log_prob(actions)
    want = Normal(mean, actor.current_std().expand_as(mean)).log_prob(
        actions).sum(dim=-1)
    assert torch.allclose(got, want, atol=0, rtol=0), \
        "log-prob must be bit-identical to the stock Normal"


def test_log_prob_of_ar1_samples_is_exact(actor):
    """AR(1) samples are scored by the SAME density they were drawn from."""
    torch.manual_seed(1)
    mean = torch.zeros(256, 17)
    dist = actor.update_distribution(mean)
    eps = actor._sample_ar1(mean)
    action = mean + dist.stddev * eps

    got = actor.get_actions_log_prob(action)
    want = Normal(mean, actor.current_std().expand_as(mean)).log_prob(
        action).sum(dim=-1)
    assert torch.allclose(got, want, atol=1e-6)


def test_sigma_clamp_is_on_the_forward_path(actor):
    """sigma above the cap is clamped for sampling, log-prob AND entropy."""
    with torch.no_grad():
        actor.std.fill_(5.0)                       # far above noise_std_max
    std = actor.current_std()
    assert torch.allclose(std, torch.full_like(std, actor.cfg.noise_std_max))

    # and the clamp has zero gradient above the bound, so the parameter stops
    # being driven upward instead of drifting while the effective sigma is pinned
    actor.std.grad = None
    actor.current_std().sum().backward()
    assert torch.count_nonzero(actor.std.grad) == 0


# ==================================================================== AR(1) ===
def test_ar1_rho_from_tau_at_50hz():
    """rho = exp(-dt/tau), per joint group, at THIS project's 50 Hz."""
    taus = {"waist": 0.30, "shoulder": 0.25, "elbow": 0.15, "wrist": 0.10}
    rho, table = build_ar1_rho(taus, 0.02)
    assert rho.shape == (17,)
    expect = {"waist": math.exp(-0.02 / 0.30), "shoulder": math.exp(-0.02 / 0.25),
              "elbow": math.exp(-0.02 / 0.15), "wrist": math.exp(-0.02 / 0.10)}
    for name, tau, r in table:
        group = next(g for g in expect if g in name)
        assert r == pytest.approx(expect[group], rel=1e-9)
    # the 60 Hz MLP values must NOT have been carried over verbatim
    assert rho[0].item() != pytest.approx(math.exp(-(1 / 60) / 0.30), rel=1e-6)


def test_ar1_tau_zero_is_white_noise():
    rho, _ = build_ar1_rho(
        {"waist": 0.0, "shoulder": 0.0, "elbow": 0.0, "wrist": 0.0}, 0.02)
    assert torch.count_nonzero(rho) == 0


def test_ar1_marginal_std_is_preserved(actor):
    """Empirical std of the AR(1) draws matches sigma to < 2 % over 10k draws.

    The variance-preserving recursion ``eps = rho*eps + sqrt(1-rho^2)*z`` has an
    exact ``N(0,1)`` marginal for any rho; this is the measurement that says the
    implementation actually has it.

    Sample size is set by the CORRELATION, not by the raw draw count: for an
    AR(1) the effective sample size is ``N * (1 - rho) / (1 + rho)``, which at
    the waist's rho = 0.9355 is only 3.3 % of N.  4096 independent chains x 64
    steps gives n_eff ~ 8.7e3 per joint, i.e. a standard error on the std of
    ~0.75 % -- comfortably inside the 2 % tolerance.  (10k *raw* draws would
    give n_eff ~ 340 and a 3.8 % standard error, so a 2 % test on them would
    fail about half the time no matter how correct the code is.)
    """
    torch.manual_seed(2)
    B, T = 4096, 64
    mean = torch.zeros(B, 17)
    actor.update_distribution(mean)
    actor._ar1_eps = None
    draws = torch.stack([actor._sample_ar1(mean) for _ in range(T)], dim=0)
    assert draws.numel() // 17 >= 10_000, "the brief's 10k-draw floor"

    emp = draws.reshape(-1, 17).std(dim=0)
    rel = (emp - 1.0).abs()
    assert float(rel.max()) < 0.02, (
        f"AR(1) marginal std off by {float(rel.max()):.4f}; per-joint {emp.tolist()}")
    assert abs(float(draws.mean())) < 0.02


def test_ar1_lag1_autocorrelation_matches_tau(actor):
    """Lag-1 autocorrelation of the noise equals ``exp(-dt/tau)`` per group."""
    torch.manual_seed(3)
    B, T = 256, 400
    mean = torch.zeros(B, 17)
    actor.update_distribution(mean)
    actor._ar1_eps = None
    draws = torch.stack([actor._sample_ar1(mean) for _ in range(T)], dim=0)

    x, y = draws[:-1], draws[1:]
    x = x - x.mean(dim=(0, 1), keepdim=True)
    y = y - y.mean(dim=(0, 1), keepdim=True)
    ac = (x * y).mean(dim=(0, 1)) / (x.std(dim=(0, 1)) * y.std(dim=(0, 1)))

    want = actor._ar1_rho
    err = (ac - want).abs()
    assert float(err.max()) < 0.03, (
        f"lag-1 autocorr {ac.tolist()} vs rho {want.tolist()}")
    # and the ordering the design intends: waist reddest, wrists whitest
    assert float(ac[0]) > float(ac[7]), "waist must be redder than the wrist"


def test_ar1_reset_redraws_rather_than_zeroing(actor):
    """A reset env gets a FRESH N(0,1) eps, not zeros.

    Zeroing would make the first post-reset action exactly the mean and leave
    eps non-stationary for several ticks -- biasing precisely the transitions an
    episode boundary already makes hardest to credit.
    """
    torch.manual_seed(4)
    mean = torch.zeros(8, 17)
    actor.update_distribution(mean)
    actor._ar1_eps = None
    actor._sample_ar1(mean)
    before = actor._ar1_eps.clone()

    dones = torch.zeros(8, dtype=torch.bool)
    dones[::2] = True
    actor.reset(dones)

    assert torch.equal(actor._ar1_eps[1::2], before[1::2]), "live envs must not move"
    assert not torch.equal(actor._ar1_eps[::2], before[::2]), "reset envs must redraw"
    assert torch.count_nonzero(actor._ar1_eps[::2]) > 0, "redraw, not zero"


def test_ar1_env_count_change_reinitialises(actor):
    actor.update_distribution(torch.zeros(8, 17))
    actor._sample_ar1(torch.zeros(8, 17))
    actor.reset(torch.zeros(16, dtype=torch.bool))
    assert actor._ar1_eps is None


# ================================================= KL / entropy magnitudes ====
def test_init_entropy_and_kl_are_in_the_documented_range(actor):
    """At init the 17-dim budgets must be the ones the MLP stack calibrated.

    ``desired_kl = 0.015`` is a SUMMED-over-17-dims budget (~8.8e-4/dim).  This
    pins two things: the analytic entropy of the initial policy, and that a
    mean shift of the size the KL budget permits is a sane fraction of sigma.
    """
    sigma = float(actor.cfg.noise_std_init)
    mean = torch.zeros(32, 17)
    actor.update_distribution(mean)

    # differential entropy of N(0, sigma) summed over 17 dims
    per_dim = 0.5 * math.log(2 * math.pi * math.e) + math.log(sigma)
    want = 17 * per_dim
    assert float(actor.entropy.mean()) == pytest.approx(want, rel=1e-5)
    # The healthy band is set by the sigma clamp, not by a magic number: a
    # per-dim entropy outside [H(noise_std_min), H(noise_std_max)] would mean
    # the head initialised outside its own admissible range.
    lo = 0.5 * math.log(2 * math.pi * math.e) + math.log(actor.cfg.noise_std_min)
    hi = 0.5 * math.log(2 * math.pi * math.e) + math.log(actor.cfg.noise_std_max)
    assert lo < per_dim < hi, (
        f"per-dim entropy {per_dim:.3f} outside [{lo:.3f}, {hi:.3f}]")
    # sigma 0.45 -> 0.620/dim -> 10.55 summed; the entropy bonus at
    # entropy_coef = 0.00175 is then ~0.018 of the loss, which is the scale the
    # MLP stack tuned that coefficient at.
    assert 9.0 < want < 12.0, f"summed entropy {want:.2f} off the 17-dim design point"

    # the per-dim mean shift that exactly spends desired_kl = 0.015
    # KL(N(m0,s) || N(m1,s)) = sum (m1-m0)^2 / (2 s^2)
    desired_kl = 0.015
    shift = math.sqrt(2 * sigma ** 2 * desired_kl / 17)
    assert 0.001 < shift < 0.05, f"KL-budget mean shift {shift:.4f} rad is odd"
    # ~0.6 % of sigma: small enough that the controller has room, large enough
    # that a single iteration is not a rounding error
    assert 0.001 < shift / sigma < 0.05


def test_kl_estimator_is_zero_for_identical_distributions(actor):
    """The updater's KL formula (rsl_rl's) must vanish on identical inputs."""
    mu = torch.randn(64, 17)
    sigma = torch.full((64, 17), 0.45)
    kl = torch.sum(
        torch.log(sigma / sigma + 1e-5)
        + (sigma.square() + (mu - mu).square()) / (2.0 * sigma.square()) - 0.5,
        dim=-1).mean()
    assert abs(float(kl)) < 2e-4
