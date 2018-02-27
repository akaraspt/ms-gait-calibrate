import pandas as pd
import numpy as np

from scipy.signal import (butter, filtfilt)

from gaitcalibrate.data.position import (BodyLocation, Position)
from gaitcalibrate.data.timeseries import Acceleration


def tilt_adjustment(acc):
    """Adjusting the tilt of accelerometer to the horizontal-vertical 
    coordinate system.
    
    Note: require prolonged walking periods for estimation

    theta_a : the angle between the horizontal plane and
              a measured anteroposterior (AP) acceleration vector.
              (positive direction being forward)
              (positive direction being upwards)
    theta_m : the angle between the horizontal plane and
              a measured mediolateral (ML) acceleration vector.
              (positive direction being acceleration to the right)
              (positive direction being upwards)

    Parameters
    ----------
    acc : Acceleration object
        Acceleration data.

    Returns
    -------
    adj_acc : Acceleration object
        Adjusted acceleration data.

    """
    sin_theta_a = np.mean(acc.data['fwd'])
    sin_theta_m = np.mean(acc.data['hor'])
    theta_a = np.arcsin(sin_theta_a)
    theta_m = np.arcsin(sin_theta_m)

    est_acc_fwd = (acc.data['fwd'] * np.cos(theta_a)) - (acc.data['ver'] * np.sin(theta_a))
    est_pro_acc_ver = (acc.data['fwd'] * np.sin(theta_a)) + (acc.data['ver'] * np.cos(theta_a))
    est_acc_hor = (acc.data['hor'] * np.cos(theta_m)) - (est_pro_acc_ver * np.sin(theta_m))
    est_acc_ver = (acc.data['hor'] * np.sin(theta_m)) + (est_pro_acc_ver * np.cos(theta_m)) - 1

    adj_acc = pd.DataFrame({
        'dt': acc.data['dt'].values,
        'ver': est_acc_ver,
        'hor': est_acc_hor,
        'fwd': est_acc_fwd
    }, columns=['dt', 'ver', 'hor', 'fwd'])

    return Acceleration(data=adj_acc,
                        fs=acc.fs)


def transform_orientation(data,
                          body_loc,
                          pos):
    ver = None
    hor = None
    fwd = None

    # Front
    if (body_loc == BodyLocation.Belt) or \
       (body_loc == BodyLocation.Chest) or \
       (body_loc == BodyLocation.Waist):
        if (pos == Position.Cen) or \
           (pos == Position.CenLeft) or \
           (pos == Position.CenRight):
            ver = data['y'].values
            hor = data['x'].values
            fwd = -data['z'].values
    # Back
    elif body_loc == BodyLocation.Waist_back or \
         body_loc == BodyLocation.Lower_back:
        if (pos == Position.Cen) or \
           (pos == Position.CenLeft) or \
           (pos == Position.CenRight):
            ver = data['y'].values
            hor = -data['x'].values
            fwd = data['z'].values

    if ver is None or hor is None or fwd is None:
        raise Exception('Invalid input monitored position.')

    transformed_acc = pd.DataFrame({
        'dt': data['dt'].values,
        'ver': ver,
        'hor': hor,
        'fwd': fwd
    }, columns=['dt', 'ver', 'hor', 'fwd'])

    return transformed_acc


def noise_filtering(acc, lowpass_f, highpass_f=None, mode="lowpass"):
    """Apply forward-backward filter on the Acceleration objects.
    If ``mode`` is set to bandpass, ``lowpass_f`` and ``highpass_f`` 
    must be specified.
    """

    # Everthing above 20 will be removed.
    # e.g., lowpass_f = 20
    if "lowpass" in mode:
        wlow = lowpass_f / acc.fnyq
        (b, a) = butter(4, wlow, btype=mode)
        filtered_df = pd.DataFrame({
            'dt': acc.data['dt'].values,
            'ver': filtfilt(b, a, acc.data['ver'].values),
            'hor': filtfilt(b, a, acc.data['hor'].values),
            'fwd': filtfilt(b, a, acc.data['fwd'].values)
        })
        filtered_acc = Acceleration(data=filtered_df,
                                    fs=acc.fs)
    # Everthing outside [0.1, 3] will be removed.
    # e.g., lowpass_f = 0.1, highpass_f = 3
    elif "bandpass" in mode:
        wlow = lowpass_f / acc.fnyq
        whigh = highpass_f / acc.fnyq
        (b, a) = butter(4, [wlow, whigh], btype=mode)
        filtered_df = pd.DataFrame({
            'dt': acc.data['dt'].values,
            'ver': filtfilt(b, a, acc.data['ver'].values),
            'hor': filtfilt(b, a, acc.data['hor'].values),
            'fwd': filtfilt(b, a, acc.data['fwd'].values)
        })
        filtered_acc = Acceleration(data=filtered_df,
                                    fs=acc.fs)
    else:
        raise Exception("Invalid mode.")

    return filtered_acc
