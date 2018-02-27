import numpy as np
import pandas as pd

from gaitcalibrate.extract.walk import extract_step
from gaitcalibrate.util.adjust_acceleration import tilt_adjustment


def estimate_walk_speed(
    acc, 
    model, 
    g2acc=True,
    n_skip_edge_step=3,
    thd_n_step_each_walk=10,
    apply_tilt_adjust=True
):
    """Estimate walking speed by applying `model` on the `acc`. 
    It assumes that `acc` is an Acceleration object containing 
    orientation-transformed, non-tilt-adjusted, un-filtered acceleration 
    data.

    Return
    ------
    `output`: esimated speed for each acceleration data point, each step, 
    and the whole `acc` object
    """

    ##################
    # Get model info #
    ##################
    estimator = model['grid_search_estimator']
    scaler = model['scaler']
    feature_ext = model['feature_ext']

    ###################
    # Tilt adjustment #
    ###################
    if apply_tilt_adjust:
        adj_acc = tilt_adjustment(acc=acc)
    else:
        adj_acc = acc

    ###################
    # Step extraction #
    ###################
    all_steps = extract_step(
        acc=adj_acc,
        g2acc=g2acc
    )

    # Remove edge steps which might not be stable
    if n_skip_edge_step > 0:
        steps = all_steps[n_skip_edge_step:-n_skip_edge_step]
        idx_steps = range(n_skip_edge_step, len(all_steps) - n_skip_edge_step)
    else:
        steps = all_steps
        idx_steps = range(len(all_steps))

    # This period is too short
    if len(steps) < thd_n_step_each_walk:
        return -1

    #################
    # Step features #
    #################
    # Feature extraction
    x = feature_ext.extract(steps)

    # Feature scaling
    scaled_x = scaler.transform(X=x)

    ############################
    # Walking speed estimation #
    ############################
    # Estimate walking speed for each step feature
    y_pred = estimator.predict(scaled_x)

    ########################################################
    # Estimated walking speed associated with acceleration #
    ########################################################
    # Walking speed associated with acceleration
    acc_spd = np.zeros(len(acc.data))
    for s, step in enumerate(all_steps):
        idx_sample = acc.data[((acc.data['dt'] >= step.data['dt'].values[0]) &
                                (acc.data['dt'] <= step.data['dt'].values[-1]))].index.values

        # Note: subtracting the first index to get around the indexing issues
        idx_sample -= acc.data.index.values[0]

        # If this step is used to estimate the walking speed, assign estimated walking speed
        if s in idx_steps:
            acc_spd[idx_sample] = y_pred[s - idx_steps[0]]
        # Otherwise, assign 0
        else:
            acc_spd[idx_sample] = 0

    ################################################
    # Estimated walking speed associated with step #
    ################################################
    # Get timestamp at the middle of each step
    mid_step_dt = np.asarray([s.data['dt'].values[len(s.data['dt'])/2] for s in steps])

    # Append zero speed at the beginning and the end to mark the beginning and end of each walk
    mid_step_dt = np.append(steps[0].data['dt'].values[0], mid_step_dt)
    mid_step_dt = np.append(mid_step_dt, steps[-1].data['dt'].values[-1])
    y_pred_ext = np.append([0], y_pred)
    y_pred_ext = np.append(y_pred_ext, [0])
    step_dt = mid_step_dt
    step_speed = y_pred_ext

    ###############################################################
    # Estimated walking speed associated with each period of walk #
    ###############################################################
    walk_start_dt = steps[0].data['dt'].values[0]
    walk_end_dt = steps[-1].data['dt'].values[-1]
    walk_speed = np.average(y_pred)

    output = {
        "acc_dt": acc.data['dt'].values,
        "acc_spd": acc_spd,
        "step_dt": step_dt,
        "step_spd": step_speed,
        "walk_start_dt": walk_start_dt,
        "walk_end_dt": walk_end_dt,
        "walk_spd": walk_speed
    }

    return output