import copy
import pytest
from scripts.train_hand_plan_dataset import validate_reference


def config(hidden=224,penalty=.1):
    return dict(dataset='/data2/eric/amortized-opt/runs/hand_plan_dataset_iid_16384_20261007',
        split_counts=dict(train=14746,validation=819,test=819),network=dict(architecture='shared_candidate',hidden=hidden),
        args=dict(steps=100000,batch=256,lr=3e-4,min_lr=3e-5,warmup=100,seed=20261010,
            weight_decay=.3,dropout=.15,gradient_l1=penalty,gradient_batch=64,
            gradient_timing_scale_s=.02,gradient_position_scale_deg=1.,
            gradient_velocity_scale_rad_s=.1,gradient_loudness_scale=.02))


def test_capacity_comparison_preserves_regularizer_and_accepts_archived_zero_default():
    assert validate_reference(config(448),config())=='scorer_width'
    archived=config(penalty=0.)
    for k in list(archived['args']):
        if k.startswith('gradient_'):del archived['args'][k]
    assert validate_reference(config(448,0.),archived)=='scorer_width'
    with pytest.raises(ValueError,match='gradient_l1'):validate_reference(config(448),config(penalty=0.))


def test_regularization_comparison_allows_only_penalty_axis_with_same_width():
    assert validate_reference(config(penalty=.1),config(penalty=0.))=='regularization'
    for key,value in [('batch',128),('seed',22),('steps',20000)]:
        changed=copy.deepcopy(config());changed['args'][key]=value
        with pytest.raises(ValueError,match=key):validate_reference(config(448),changed)
    changed=config();changed['split_counts']['train']=99
    with pytest.raises(ValueError,match='dataset/splits'):validate_reference(config(448),changed)
