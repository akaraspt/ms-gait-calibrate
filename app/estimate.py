import functools
import pickle
import ntpath
import os
import shutil

from multiprocessing.dummy import Pool as ThreadPool

import numpy as np

from dateutil import rrule

from app import app
from gaitcalibrate.extract.walk import extract_walk_csv
from gaitcalibrate.estimate.speed import estimate_walk_speed


def estimate_speed(selected_file, selected_model):
    # Output directory
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], selected_file)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    ##############
    # Load model #
    ##############
    with open(os.path.join(app.config['MODEL_FOLDER'], selected_model), 'rb') as f:
        model = pickle.load(f)
        sampling_rate = model['sampling_rate']
        body_location = model['body_location']
        position = model['position']

    #####################
    # Load acceleration #
    #####################
    csv_file = os.path.join(app.config['UPLOAD_FOLDER'], selected_file)
    walks = extract_walk_csv(
        csv_file=csv_file,
        n_skip_edge_step=2,
        thd_n_step_each_walk=3,
        sampling_rate=sampling_rate,
        body_location=body_location,
        position=position
    )

    if len(walks) == 0:
        print "There is no walk to estimate."
        return {
            "walks": [],
            "outputs": [],
            "output_dir": output_dir
        }

    ##########################
    # Estimate walking speed #
    ##########################
    # Worker pool for multithreading
    pool = ThreadPool(8)

    # Specify arguments for estimate_walk_speed
    func = functools.partial(
        estimate_walk_speed,
        model=model,
        g2acc=True,
        n_skip_edge_step=2,
        thd_n_step_each_walk=3
    )

    # Estimate speed for each walk
    # Parallel map call - it blocks until the result is ready.
    outputs = pool.map(func, walks)

    # Save to an NPY file
    filename = ntpath.basename(selected_file)
    filename = filename.split(".")[0]
    np.save(
        os.path.join(output_dir, "{}.npy".format(filename)), 
        np.asarray(outputs)
    )

    return {
        "walks": walks,
        "outputs": outputs,
        "output_dir": output_dir
    }


def bufcount(filename):
    f = open(filename)                  
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.read # loop optimization

    buf = read_f(buf_size)
    while buf:
        lines += buf.count('\n')
        buf = read_f(buf_size)

    return lines


def hour_range(start_dt, end_dt, inc_start=True, inc_end=True):
    if inc_start:
        yield start_dt

    rule = rrule.rrule(rrule.HOURLY, byminute=0, bysecond=0, dtstart=start_dt)
    for x in rule.between(start_dt, end_dt, inc=False):
        yield x

    if inc_end:
        yield end_dt
