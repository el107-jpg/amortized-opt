import copy
import hashlib
import json
import pytest
from scripts.train_hand_plan_dataset import load_reference_evaluation


def fixture(tmp_path):
    snapshot=tmp_path/'old_weights.pt';snapshot.write_bytes(b'frozen old policy')
    runtime=dict(rollout_runtime_version='matched_decoder_speed_contact_reset_v2',
        scene_reset_protocol='fresh_process_scene_single_wave_v1')
    config=dict(dataset=str(tmp_path/'dataset'),split_counts=dict(validation=819,test=819),rollout_runtime=runtime)
    reference=dict(format='frozen_hand_plan_reference_v1',dataset=config['dataset'],
        rollout_runtime=runtime,model_training_runtime=dict(rollout_runtime_version='legacy_contact_reset_v0'),
        heldout_inputs_match=True,held_out_used_for_training=False,weights_selected_by_evaluation=False,
        checkpoint_snapshot=str(snapshot),reference_weights_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        metrics=dict(validation=dict(cases=819),test=dict(cases=819)))
    path=tmp_path/'reference.json';path.write_text(json.dumps(reference))
    return path,config,reference,snapshot


def test_old_policy_can_be_read_only_evaluated_on_matching_new_label_runtime(tmp_path):
    path,config,reference,_=fixture(tmp_path)
    assert load_reference_evaluation(path,config)['model_training_runtime']['rollout_runtime_version']=='legacy_contact_reset_v0'
    reference['rollout_runtime']=dict(rollout_runtime_version='legacy_contact_reset_v0')
    path.write_text(json.dumps(reference))
    with pytest.raises(ValueError,match='runtime differs'):load_reference_evaluation(path,config)


def test_frozen_reference_rejects_changed_inputs_labels_weights_or_training_use(tmp_path):
    path,config,reference,snapshot=fixture(tmp_path)
    for key,value in [('heldout_inputs_match',False),('held_out_used_for_training',True),('weights_selected_by_evaluation',True)]:
        changed=copy.deepcopy(reference);changed[key]=value;path.write_text(json.dumps(changed))
        with pytest.raises(ValueError,match='read-only'):load_reference_evaluation(path,config)
    path.write_text(json.dumps(reference));snapshot.write_bytes(b'changed policy')
    with pytest.raises(ValueError,match='weights changed'):load_reference_evaluation(path,config)
