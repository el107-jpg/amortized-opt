import numpy as np

from scripts.reclassify_decoder_collision_table import select_candidates


def test_tighter_collision_classification_searches_all_durations_and_both_yaw_hints():
    data=dict(destination=np.array([1,1,1,1,2,3]),sample=np.zeros(6,int),
        segments=np.array([10,30,20,20,50,75]),yaw_hint_index=np.array([0,0,1,0,0,0]),
        relaxed_non_torque_success=np.ones(6,bool),
        collision_min_sdf=np.array([-.006,-.004,-.003,-.003,0.,0.]),
        terminal_region_dist=np.array([0.,0.,0.,0.,.011,0.]),
        torque_excess_average_Nm=np.array([1.,1.,4.,4.,1.,2.]))
    best,passing=select_candidates(data,.005,.01,5.)
    assert best=={(1,0):3,(3,0):5}
    np.testing.assert_array_equal(passing,[False,True,True,True,False,True])
    assert data['segments'][best[1,0]]==20
    assert data['segments'][best[3,0]]==75  # Deadline longer than 1 s is retained.
