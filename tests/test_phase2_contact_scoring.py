from amortized_opt.cost import CostConfig
from amortized_opt.train_cost import phase2_cost_config
from scripts.train_phase2 import _resolve_contact_scoring_mode
from scripts.train_terminal_pose import resolve_terminal_contact_mode


def test_new_phase2_uses_v5_but_archived_resumes_preserve_their_objective():
    assert CostConfig().contact_scoring_mode=='legacy'
    assert phase2_cost_config().contact_scoring_mode=='isaac_v5'
    old={'train_cfg':{'cost':{'target_top_entry_depth_m':None}}}
    newer={'train_cfg':{'cost':{'contact_scoring_mode':'isaac_v4'}}}
    assert _resolve_contact_scoring_mode('auto',None,resume=False)=='isaac_v5'
    assert _resolve_contact_scoring_mode('auto',old,resume=True)=='legacy'
    assert _resolve_contact_scoring_mode('auto',newer,resume=True)=='isaac_v4'
    assert _resolve_contact_scoring_mode('isaac_v4',old,resume=True)=='isaac_v4'
    assert resolve_terminal_contact_mode('auto',None)=='isaac_v5'
    assert resolve_terminal_contact_mode('auto',{'config':{'w_com':1.}})=='legacy'
    assert resolve_terminal_contact_mode('auto',{'config':{'contact_scoring_mode':'isaac_v4'}})=='isaac_v4'
