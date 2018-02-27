import ntpath
import os
import re

import numpy as np
import pandas as pd

from sklearn import preprocessing

from gaitcalibrate import dt_format
from gaitcalibrate.data.position import BodyLocation
from gaitcalibrate.data.timeseries import Acceleration
from gaitcalibrate.extract.footcontact import FootContactPeakZCPtoM
from gaitcalibrate.extract.step import extract_step
from gaitcalibrate.util.adjust_acceleration import transform_orientation, tilt_adjustment
from gaitcalibrate.util.helper import get_list_seq


class WalkScaledRestLowVar(object):

    def __init__(self,
                 window,
                 step_window,
                 thd_var_rest_window,
                 thd_walk_window):
        self.window = window
        self.step_window = step_window
        self.thd_var_rest_window = thd_var_rest_window
        self.thd_walk_window = thd_walk_window

    def get_walk(self, acc):
        idx_rest_sample = np.empty(0, dtype=int)

        # Scaling each dimension
        scaled_ver = preprocessing.scale(acc.data['ver'].values)
        scaled_hor = preprocessing.scale(acc.data['hor'].values)
        scaled_fwd = preprocessing.scale(acc.data['fwd'].values)

        # Get resting periods
        for w in range(0, acc.size-self.window+1, self.step_window):
            start_idx = w
            end_idx = start_idx + self.window

            if (np.var(scaled_ver[start_idx:end_idx]) < self.thd_var_rest_window) and \
               (np.var(scaled_hor[start_idx:end_idx]) < self.thd_var_rest_window) and \
               (np.var(scaled_fwd[start_idx:end_idx]) < self.thd_var_rest_window):
                idx_rest_sample = np.union1d(idx_rest_sample, np.arange(start_idx, end_idx+1))

        # Get indices of walking periods
        idx_walk_sample = np.setdiff1d(np.arange(acc.size), idx_rest_sample)
        list_idx_walk = get_list_seq(idx_walk_sample)

        # Get acceleration of each walk
        walks = np.empty(0, dtype=object)
        for idx_w, w in enumerate(list_idx_walk):
            if len(w) > self.thd_walk_window:
                walk_acc = acc.get_idx(w)
                walks = np.append(walks, walk_acc)

        return walks


def get_walk_extractor(body_loc):
    """Get a walk extractor based on the device attached locations."""

    # At belt
    if body_loc == BodyLocation.Belt:
        walk_ext = WalkScaledRestLowVar(window=100,
                                        step_window=25,
                                        thd_var_rest_window=5e-2,
                                        thd_walk_window=400)
    # At chest
    elif body_loc == BodyLocation.Chest:
        walk_ext = WalkScaledRestLowVar(window=100,
                                       step_window=25,
                                       thd_var_rest_window=5e-2,
                                       thd_walk_window=400)
    # At front waist - at the top of the hip
    elif body_loc == BodyLocation.Waist:
        walk_ext = WalkScaledRestLowVar(window=100,
                                       step_window=25,
                                       thd_var_rest_window=5e-2,
                                       thd_walk_window=400)
    # At back waist
    elif body_loc == BodyLocation.Waist_back:
        walk_ext = WalkScaledRestLowVar(window=100,
                                       step_window=25,
                                       thd_var_rest_window=5e-2,
                                       thd_walk_window=400)
    # At lower back - 3 fingers below the top of the hip
    elif body_loc == BodyLocation.Lower_back:
        walk_ext = WalkScaledRestLowVar(window=100,
                                       step_window=25,
                                       thd_var_rest_window=5e-2,
                                       thd_walk_window=400)
    else:
        raise Exception('Invalid body location.')

    return walk_ext


def extract_walk_csv(
    csv_file, 
    n_skip_edge_step=2, 
    thd_n_step_each_walk=10,
    output_dir=None,
    sampling_rate=100.0,
    body_location="lower_back",
    position="center_right",
    apply_tilt_adjust=False,
):
    """Extract walk from the unprocessed CSV file. Walk data will
    be orientation-transformed and stored in Acceleration objects.
    Each object contains acceleration data in terms of vertical,
    horizontal and forward axes. 
    
    The CSV file with metadata will have the first line as follows:
    - line 1: sampling_rate,body_location,position[,walking_speed]
    - line 2: timestamp,x,y,z
    - line 3: timestamp,x,y,z
    - ...

    If the CSV file does not have metadata, it either uses the default
    values or the ones specified by the caller.

    Return
    ------
    - If `output_dir` is not specified, return a list of Acceleration objects.
    - Otherwise, it saves a list of walks in an NPY file in the `output_dir`
    """

    print "Extracting walks from: {}".format(csv_file)

    walks = []

    ############
    # Metadata #
    ############
    # Read the first line to get the metadata
    with open(csv_file, 'r') as f:
        first_line = f.readline()
        if re.search('[a-zA-Z]', first_line) is not None:
            cols = first_line.split(",")
            sampling_rate = float(cols[0].strip())
            body_location = cols[1].strip()
            position = cols[2].strip()

            # Skip metadata line
            skiprows = 1
        else:
            skiprows = 0

    print "{} (meta={}): {}, {}, {}".format(
        csv_file, 
        skiprows == 1,
        sampling_rate, 
        body_location, 
        position)

    #################################
    # Create an Acceleration object #
    #################################
    # Function to parse datetime format
    def dateparse(x): return pd.datetime.strptime(x, dt_format)
    acc = pd.read_csv(
        csv_file,
        names=['dt', 'x', 'y', 'z'],
        header=None,
        parse_dates=['dt'],
        date_parser=dateparse,
        skiprows=skiprows
    )
    # 3-D acceleration will be set according to the orientation of the device attached
    # to the human body
    transformed_acc = transform_orientation(data=acc,
                                            body_loc=body_location,
                                            pos=position)
    acc = Acceleration(data=transformed_acc,
                       fs=sampling_rate)

    ###################
    # Walk extraction #
    ###################
    walk_ext = get_walk_extractor(body_loc=body_location)
    all_walks = walk_ext.get_walk(acc=acc)

    for w in all_walks:
        # Tile adjustment --> might be better to use bandpass
        adj_w = tilt_adjustment(acc=w)

        # Extract steps
        steps = extract_step(acc=adj_w)
        
        # Remove edge steps which might not be stable
        if n_skip_edge_step > 0:
            steps = steps[n_skip_edge_step:-n_skip_edge_step]

        # Check whether the number of steps are sufficient to do walking speed estimation
        if len(steps) > thd_n_step_each_walk:
            if apply_tilt_adjust:
                walks.append(adj_w)
            else:
                walks.append(w)

    # If output_dir is not specified, return walks
    if output_dir is None:
        return walks
    # Otherwise, save walks in the file
    else:
        # Get input filename
        filename = ntpath.basename(csv_file)
        filename = filename.split(".")[0]

        # Save to an NPY file
        walks = np.asarray(walks)
        np.save(
            os.path.join(output_dir, "{}.npy".format(filename)), 
            walks
        )
