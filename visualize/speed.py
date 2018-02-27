import argparse
import ntpath
import os

import numpy as np
import pandas as pd

from bokeh.io import output_file, show
from bokeh.layouts import column
from bokeh.plotting import figure

from gaitcalibrate.data.timeseries import Acceleration
from gaitcalibrate.extract.walk import extract_walk_csv
from gaitcalibrate.util.adjust_acceleration import tilt_adjustment, noise_filtering


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acc_file", type=str, required=True,
                       help="File path to the CSV or NPY file that contains walking data.")
    parser.add_argument("--speed_file", type=str, required=True,
                       help="File path to the NPY file containing estimate walking speeds.")
    parser.add_argument("--output_dir", type=str, default="outputs/speed_plots",
                       help="Directory where to save outputs.")
    parser.add_argument("--sampling_rate", type=float, default=100.0,
                       help="Directory where to save outputs.")
    parser.add_argument("--body_location", type=str, default="lower_back",
                       help="Directory where to save outputs.")
    parser.add_argument("--position", type=str, default="center_right",
                       help="Directory where to save outputs.")
    args = parser.parse_args()

    # Get input filename
    filename = ntpath.basename(args.acc_file)
    filename = filename.split(".")[0]

    # Output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # CSV file
    if args.acc_file.lower().endswith('.csv'):
        walks = extract_walk_csv(
            csv_file=args.acc_file,
            n_skip_edge_step=2,
            thd_n_step_each_walk=10,
            sampling_rate=args.sampling_rate,
            body_location=args.body_location,
            position=args.position
        )
    # NPY file - contain a list of walks
    elif args.acc_file.lower().endswith('.npy'):
        walks = np.load(args.acc_file)
    else:
        raise Exception("Invalid acc_file.")
        
    speeds = np.load(args.speed_file)

    assert len(walks) == len(speeds)

    # Create subdirectory
    output_plot_dir = os.path.join(args.output_dir, filename)
    if not os.path.exists(output_plot_dir):
        os.makedirs(output_plot_dir)

    # Using Bokeh to create html plots
    figs = []
    for i in range(len(walks)):
        # Tilt Adjustment and Noise Filtering
        acc = walks[i]
        acc = tilt_adjustment(acc=acc)
        acc = noise_filtering(
            acc=adj_acc,
            lowpass_f=20,
            mode="lowpass"
        )
        
        # Acceleration and Speed
        acc = acc.data
        speed = speeds[i]

        # Html file for each walk
        output_file(os.path.join(output_plot_dir, "{}.html".format(i)))

        fig = figure(
            x_axis_type="datetime", 
            title="Acceleration", 
            plot_width=1000,
            plot_height=600,
            tools="hover,crosshair,pan,wheel_zoom,box_zoom,undo,redo,reset,"
        )
        fig.grid.grid_line_alpha=0.5
        fig.xaxis.axis_label = 'Time'
        fig.yaxis.axis_label = 'Acceleration (g) or Walking Speed (m/s)'

        # Acceleration
        fig.line(acc['dt'], acc['ver'], color='red', legend='ver', alpha=0.3)
        fig.line(acc['dt'], acc['hor'], color='green', legend='hor', alpha=0.3)
        fig.line(acc['dt'], acc['fwd'], color='blue', legend='fwd', alpha=0.3)
        # Speed
        fig.circle(speed['step_dt'], speed['step_spd'], color="black", legend="spd")
        fig.line(speed['step_dt'], speed['step_spd'], color='black', legend='spd', alpha=0.5)

        fig.legend.location = "top_right"

        figs.append(fig)

    # Show the results
    show(column(*figs))


if __name__ == "__main__":
    main()