import datetime
import os
from shutil import copyfile

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import dates

from app import app
from gaitcalibrate import (fig_width, fig_height, fig_save_dpi, dt_format)
from gaitcalibrate.extract.walk import get_walk_extractor
from gaitcalibrate.io.csv import CSVAccelerationReader, CSVSpeedReader
from gaitcalibrate.util.adjust_acceleration import tilt_adjustment


def create_walking_graphs(filepath, sampling_rate, body_loc, pos):
    # Initialization
    acc_reader = CSVAccelerationReader()
    walk_ext = get_walk_extractor(body_loc=body_loc)

    # Get acceleration only in the period of training
    # (i.e. exclude the acceleration that are the walk to and from the experiment space)
    acc = acc_reader.load(filepath=filepath,
                          body_loc=body_loc,
                          pos=pos,
                          fs=sampling_rate)
    # Walk extraction
    walks = walk_ext.get_walk(acc=acc)

    # This figure is used to identify the index of selected walks
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=fig_save_dpi)
    
    #####################
    # Acceleration data #
    #####################
    ax = plt.subplot2grid((3,1), (0,0), rowspan=2)
    line_ver, = plt.plot_date(acc.data['dt'].values, acc.data['ver'].values,
                              fmt='-r', xdate=True, ydate=False, label='ver')
    line_hor, = plt.plot_date(acc.data['dt'].values, acc.data['hor'].values,
                              fmt='-g', xdate=True, ydate=False, label='hor')
    line_fwd, = plt.plot_date(acc.data['dt'].values, acc.data['fwd'].values,
                              fmt='-b', xdate=True, ydate=False, label='fwd')
    hfmt = dates.DateFormatter('%H:%M')
    ax = plt.gca()
    dt_range = pd.to_datetime(acc.data['dt'].values[-1]) - pd.to_datetime(acc.data['dt'].values[0])
    if dt_range > datetime.timedelta(minutes=60):
        ax.xaxis.set_major_locator(dates.MinuteLocator(interval=5))
    else:
        ax.xaxis.set_major_locator(dates.MinuteLocator())
    ax.xaxis.set_major_formatter(hfmt)
    plt.xticks(rotation='vertical')
    plt.ylabel('Acceleration ($g$)', fontsize=18)
    plt.xlabel('Time of day', fontsize=18)
    plt.title('Acceleration data and a plot that indicates detected walking periods', fontsize=20)
    plt.legend(handles=[line_ver, line_hor, line_fwd],
               loc='center left', bbox_to_anchor=(1, 0.5))
    plt.axis('tight')
    plt.ylim([-2.5, 2.5])

    #########################
    # Walking period labels #
    #########################
    annotates = []
    ax = plt.subplot2grid((3,1), (2,0))
    walk_sample = np.empty(0, dtype=int)
    for idx_w, walk in enumerate(walks):
        if len(walk.data.index.values) > sampling_rate * 2:
            # Note: subtracting the first index to get around the indexing issues
            tmp_sample = walk.data.index.values - acc.data.index.values[0]
            walk_sample = np.append(walk_sample, tmp_sample)
            mid_dt = walk.data['dt'].values[len(walk.data['dt'].values)/2]
            mid_dt = pd.to_datetime(mid_dt)
            annotates.append((idx_w, mid_dt))
    walk_label = np.zeros(len(acc.data))
    walk_label[walk_sample] = 1
    line_walk, = plt.plot_date(acc.data['dt'].values, walk_label,
                               fmt='-k', xdate=True, ydate=False, label='walk')
    hfmt = dates.DateFormatter('%H:%M')
    ax = plt.gca()
    dt_range = pd.to_datetime(acc.data['dt'].values[-1]) - pd.to_datetime(acc.data['dt'].values[0])
    if dt_range > datetime.timedelta(minutes=60):
        ax.xaxis.set_major_locator(dates.MinuteLocator(interval=5))
    else:
        ax.xaxis.set_major_locator(dates.MinuteLocator())
    ax.xaxis.set_major_formatter(hfmt)
    plt.xticks(rotation='vertical')
    plt.yticks([0, 1], ['0', '1'])
    plt.ylabel('Walking period', fontsize=18)
    plt.xlabel('Time of day', fontsize=18)
    plt.legend(handles=[line_walk],
               loc='center left', bbox_to_anchor=(1, 0.5))
    for ann in annotates:
        ax.annotate(
            str(ann[0]), 
            xy=(dates.date2num(ann[1]), 1), 
            xytext=(dates.date2num(ann[1]), 1.75),
            ha="center", va="center",
            arrowprops=dict(facecolor='black', shrink=0.05),
        )
    plt.axis('tight')
    plt.ylim([0, 2.5])
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    figpath = os.path.join(app.config['TMP_FOLDER'], 'all_walk_detect.png')
    try:
        os.remove(figpath)
    except OSError:
        pass
    plt.savefig(figpath)
    plt.close('all')

    return figpath

def generate_train_data(filepath, sampling_rate, body_loc, pos, selected_walk_idx, selected_walk_spd):
    # Initialization
    acc_reader = CSVAccelerationReader()
    walk_ext = get_walk_extractor(body_loc=body_loc)

    # Get acceleration only in the period of training
    # (i.e. exclude the acceleration that are the walk to and from the experiment space)
    acc = acc_reader.load(filepath=filepath,
                          body_loc=body_loc,
                          pos=pos,
                          fs=sampling_rate)
    # Walk extraction
    walks = walk_ext.get_walk(acc=acc)

    # Select walk periods
    walks = walks[selected_walk_idx]

    # Assign walk speed for each walk
    spd = np.zeros(len(acc.data), dtype=float)
    for idx_w, walk in enumerate(walks):
        # Note: subtracting the first index to get around the indexing issues
        spd_idx = walk.data.index.values - acc.data.index.values[0]
        spd[spd_idx] = selected_walk_spd[idx_w]

    plt.figure(figsize=(fig_width, fig_height), dpi=fig_save_dpi)
    #####################
    # Acceleration data #
    #####################
    ax = plt.subplot2grid((3,1), (0,0), rowspan=2)
    line_ver, = plt.plot_date(acc.data['dt'].values, acc.data['ver'].values,
                              fmt='-r', xdate=True, ydate=False, label='ver')
    line_hor, = plt.plot_date(acc.data['dt'].values, acc.data['hor'].values,
                              fmt='-g', xdate=True, ydate=False, label='hor')
    line_fwd, = plt.plot_date(acc.data['dt'].values, acc.data['fwd'].values,
                              fmt='-b', xdate=True, ydate=False, label='fwd')
    from matplotlib import dates
    hfmt = dates.DateFormatter('%H:%M')
    ax = plt.gca()
    dt_range = pd.to_datetime(acc.data['dt'].values[-1]) - pd.to_datetime(acc.data['dt'].values[0])
    if dt_range > datetime.timedelta(minutes=60):
        ax.xaxis.set_major_locator(dates.MinuteLocator(interval=5))
    else:
        ax.xaxis.set_major_locator(dates.MinuteLocator())
    ax.xaxis.set_major_formatter(hfmt)
    plt.xticks(rotation='vertical')
    plt.ylabel('Acceleration ($g$)', fontsize=18)
    plt.xlabel('Time of day', fontsize=18)
    plt.title('Acceleration data and their corresponding walking speeds (only for selected walking periods)', fontsize=20)
    plt.legend(handles=[line_ver, line_hor, line_fwd],
               loc='center left', bbox_to_anchor=(1, 0.5))
    plt.axis('tight')

    ######################
    # Walking speed data #
    ######################
    ax = plt.subplot2grid((3,1), (2,0))
    line_spd, = plt.plot_date(acc.data['dt'].values, spd,
                              fmt='-c', xdate=True, ydate=False, label='spd')
    hfmt = dates.DateFormatter('%H:%M')
    ax = plt.gca()
    dt_range = pd.to_datetime(acc.data['dt'].values[-1]) - pd.to_datetime(acc.data['dt'].values[0])
    if dt_range > datetime.timedelta(minutes=60):
        ax.xaxis.set_major_locator(dates.MinuteLocator(interval=5))
    else:
        ax.xaxis.set_major_locator(dates.MinuteLocator())
    ax.xaxis.set_major_formatter(hfmt)
    plt.xticks(rotation='vertical')
    plt.legend(handles=[line_spd],
               loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel('Walking speed ($m/s$)', fontsize=18)
    plt.xlabel('Time of day', fontsize=18)
    plt.axis('tight')
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.ylim([0, 2])
    figpath = os.path.join(app.config['TMP_FOLDER'], 'selected_walk_speed.png')
    try:
        os.remove(figpath)
    except OSError:
        pass
    plt.savefig(figpath)
    plt.close('all')

    # Get training directory
    src_fname = os.path.basename(filepath)
    src_fname = src_fname.split(".")[0]
    train_dir = os.path.join(app.config['UPLOAD_FOLDER'], src_fname)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    # Get calibration directory to store another copy for calibration
    calib_dir = app.config['CALIBRATE_FOLDER']
    if not os.path.exists(calib_dir):
        os.makedirs(calib_dir)

    # Create walking speed files
    output = np.empty((len(spd), 2), dtype=object)
    output[:, 0] = np.asarray([_dt.strftime(dt_format) for _dt in acc.data['dt']])
    output[:, 1] = spd
    spd_filepath = os.path.join(train_dir, 'speed_' + src_fname + '.csv')
    np.savetxt(spd_filepath, output, fmt="%s,%f")

    # Reload the walking speed data
    spd_reader = CSVSpeedReader()
    spd = spd_reader.load(filepath=spd_filepath)

    # Extracting acceleration data with speed
    # It also performs tilt adjustment for each walk, and then save to files
    for w in range(len(walks)):
        walk = walks[w]

        # Tile adjustment
        adj_w = tilt_adjustment(acc=walk)

        # Compute average speed
        raw_speed = spd.get_dt(start_dt=walk.data['dt'].values[0],
                                end_dt=walk.data['dt'].values[-1])
        avg_speed = np.mean(raw_speed.data['spd'].values)

        # Save each walk to file
        output = np.empty((len(adj_w.data), 5), dtype=object)
        output[:, 0] = np.asarray([_dt.strftime(dt_format) for _dt in adj_w.data['dt']])
        output[:, 1] = adj_w.data['ver'].values
        output[:, 2] = adj_w.data['hor'].values
        output[:, 3] = adj_w.data['fwd'].values
        output[:, 4] = avg_speed
        walkspeed_path = os.path.join(train_dir, 'walkspeed_{}_{}.csv'.format(src_fname, w))
        np.savetxt(walkspeed_path, output, fmt="%s,%f,%f,%f,%f")

        # Add metadata into the first line of the file
        # Note: sampling rate, body location, position, and average walking speed
        meta_walkspeed_path = os.path.join(train_dir, 'meta_walkspeed_{}_{}.csv'.format(src_fname, w))
        if os.path.exists(walkspeed_path):
            with open(walkspeed_path, 'r') as f_read:
                with open(meta_walkspeed_path, 'wb') as f_write:
                    f_write.write('{},{},{},{}\n'.format(
                        sampling_rate,
                        body_loc,
                        pos,
                        avg_speed
                    ))
                    f_write.write(f_read.read())
            try:
                os.remove(walkspeed_path)
            except OSError:
                pass

            try:
                src = meta_walkspeed_path
                dst = os.path.join(calib_dir, 'meta_walkspeed_{}_{}.csv'.format(src_fname, w))
                copyfile(src, dst)
            except IOError:
                raise Exception("Cannot copy a file to calibration directory")

    return figpath
