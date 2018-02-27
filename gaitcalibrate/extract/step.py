import numpy as np
import pandas as pd

from gaitcalibrate import g
from gaitcalibrate.data.timeseries import Acceleration
from gaitcalibrate.extract.footcontact import FootContactPeakZCPtoM

from gaitcalibrate.util.helper import gen_idx_range
from gaitcalibrate.util.adjust_acceleration import noise_filtering


def extract_step(acc, g2acc=True):
    """Extract walk from an Acceleration object."""

    ###################
    # Noise filtering #
    ###################
    f_acc = noise_filtering(
        acc=acc, 
        lowpass_f=20,
        mode="lowpass"
    )

    ##########################
    # Foot-contact detection #
    ##########################
    # Further filter the forward acceleration for foot-contact detection
    f_acc_fwd_fc = noise_filtering(
        acc=acc, 
        lowpass_f=0.1,
        highpass_f=6,
        mode="bandpass"
    ).data['fwd'].values
    # f_acc_fwd_fc = noise_filtering(
    #     acc=acc, 
    #     lowpass_f=5,
    #     mode="lowpass"
    # ).data['fwd'].values

    # Get foot-contact for walking speed estimation
    fc_detector = FootContactPeakZCPtoM(min_peak=0.01)
    peak_zc_down, zc_ptom, peak = fc_detector.get_footcontact(x=f_acc_fwd_fc)
    idx_fc = peak_zc_down

    # No steps detected
    if len(idx_fc) == 0:
        return []

    # Count number of steps for this walk
    n_steps = len(idx_fc)

    ###############################################
    # Convert unit from g to acceleration (m/s^2) #
    ###############################################
    if g2acc:
        data = pd.DataFrame({
            'dt': f_acc.data['dt'].values,
            'ver': f_acc.data['ver'].values * g,
            'hor': f_acc.data['hor'].values * g,
            'fwd': f_acc.data['fwd'].values * g
        }, columns=['dt', 'ver', 'hor', 'fwd'])
        acc = Acceleration(data=data,
                           fs=f_acc.fs)
    else:
        acc = f_acc

    ###################
    # Step extraction #
    ###################
    idx_sample = gen_idx_range(idx_fc, len(acc.data))
    steps = np.empty(len(idx_sample), dtype=object)
    for s_idx in xrange(len(steps)):
        idx_step = idx_sample[s_idx]
        steps[s_idx] = acc.get_idx(idx_step)

    # Sanity check
    if n_steps != (len(steps) - 1) or n_steps != len(idx_fc):
        raise Exception('Incorrect step extraction.')
    
    return steps


def get_steps(acc,
              fc_idx):
    idx_sample = gen_idx_range(fc_idx, acc.size)
    steps = np.empty(idx_sample.size, dtype=object)
    for s in range(steps.size):
        idx_step = idx_sample[s]
        steps[s] = acc.get_idx(idx_step)
