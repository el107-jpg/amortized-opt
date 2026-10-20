import numpy as np
import torch
from isaac_rl.hand_plan_large_dataset import collection_tasks,task_splits,heldout_inputs
from isaac_rl.hand_plan_ranker import encode_clip,cartesian_tips
from amortized_opt.constants import ready_joint_pos
from tests.test_neural_hand_plans import clip


def test_exact_training_budget_and_fixed_heldout_counts_include_partial_batches():
    tasks=collection_tasks(100000,256,dict(validation=819,test=819));splits=task_splits(tasks)
    assert len(tasks)==399 and tasks[390]['count']==160
    assert {k:len(v) for k,v in splits.items()}==dict(train=100000,validation=819,test=819)
    np.testing.assert_array_equal(np.sort(np.concatenate(list(splits.values()))),np.arange(101638))


def test_manifest_heldouts_reuse_inputs_in_original_order_and_ignore_outcomes(tmp_path):
    import json
    folders=[]
    for batch_index in range(2):
        p=tmp_path/f'source{batch_index}';(p/'batches').mkdir(parents=True);folders.append(str(p))
        panel=[clip(),clip()];q=np.asarray(ready_joint_pos(),np.float32)
        states=[dict(q=q.tolist(),qd=np.zeros(17).tolist(),tips=cartesian_tips(q).tolist(),initial_drums=[-1,-1]) for _ in panel]
        obs=np.stack([encode_clip(c,**s) for c,s in zip(panel,states)])
        torch.save(dict(batch=dict(panel=panel,states=states,observations=obs),outcomes='Never read this label payload'),p/'batches/batch_00001.pt')
    combined=tmp_path/'combined';combined.mkdir();(combined/'manifest.json').write_text(json.dumps(dict(source_datasets=folders)))
    torch.save(dict(train=np.array([0,2]),validation=np.array([3]),test=np.array([1])),combined/'splits.pt')
    got=heldout_inputs(combined,None)
    assert got['validation']['source_case_ids'].tolist()==[3] and got['test']['source_case_ids'].tolist()==[1]
    np.testing.assert_array_equal(got['validation']['observations'][0],encode_clip(clip()))
