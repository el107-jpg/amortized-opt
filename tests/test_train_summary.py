import pytest

from amortized_opt.train_cost import _invocation_throughput


def test_resume_throughput_uses_only_current_invocation_updates():
    rates = _invocation_throughput(
        updates_attempted=300,
        updates_executed=294,
        wall_s=1000.0,
    )
    assert rates["updates_attempted"] == 300
    assert rates["updates_executed"] == 294
    assert rates["steps_per_s"] == pytest.approx(0.3)
    assert rates["executed_steps_per_s"] == pytest.approx(0.294)
    # A resumed absolute step (e.g. 14_900) is deliberately not an input.
