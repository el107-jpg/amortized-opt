import numpy as np

from scripts.report_decoder_torque_integrals import threshold_support


def test_threshold_curve_counts_departures_once_across_deadlines_and_yaw_hints():
    data=dict(hand=np.array([0,0,0,1,0]),source=np.array([1,1,1,3,1]),
        destination=np.array([2,2,3,4,2]),sample=np.array([0,0,0,1,2]),
        torque_excess_average_Nm=np.array([2.,.2,.1,5.1,.3]),
        relaxed_non_torque_success=np.array([True,True,False,True,True]))
    # An additional yaw/deadline for sample 0 cannot add another supported start.
    duplicate={key:value[:1].copy() for key,value in data.items()}
    duplicate['torque_excess_average_Nm'][0]=.21
    assert threshold_support([data,duplicate],.25)==dict(threshold_average_excess_Nm=.25,
        supported_pairs=1,supported_starts=1)
    assert threshold_support([data,duplicate],1.)==dict(threshold_average_excess_Nm=1.,
        supported_pairs=1,supported_starts=2)
    assert threshold_support([data,duplicate],5.2)==dict(threshold_average_excess_Nm=5.2,
        supported_pairs=2,supported_starts=3)
