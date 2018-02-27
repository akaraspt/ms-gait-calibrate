import argparse
import datetime
import functools
import glob
import os
import shutil

from multiprocessing.dummy import Pool as ThreadPool

import numpy as np

from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file

from gaitcalibrate.extract.walk import extract_walk_csv


def plot_graph_html(acc, output_file):
    """Plot an Acceleration object into a graph in a HTML file."""

    fig = figure(
        x_axis_type="datetime", 
        title="Acceleration",
        plot_width=1200,
        plot_height=600,
        tools="hover,crosshair,pan,wheel_zoom,box_zoom,undo,redo,reset,"
    )
    fig.grid.grid_line_alpha = 0.5
    fig.xaxis.axis_label = 'Time'
    fig.yaxis.axis_label = 'Acceleration (g)'
    
    fig.line(acc.data['dt'], acc.data['fwd'], color='red', legend='fwd')
    fig.line(acc.data['dt'], acc.data['hor'], color='green', legend='hor')
    fig.line(acc.data['dt'], acc.data['ver'], color='blue', legend='ver')
    fig.legend.location = "top_right"

    output_file(output_file, title="Accerelation")

    show(gridplot([[fig]]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                       help="File path to the CSV file.")
    parser.add_argument("--output_dir", type=str, default="outputs/walks",
                       help="Directory where to save outputs.")
    args = parser.parse_args()

    # Output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    # Get list of all csv files
    csv_files = [i for i in glob.glob(os.path.join(args.data_dir, '*.csv'))]

    # Worker pool for multithreading
    pool = ThreadPool(8)

    # Specify arguments for extract_walk_csv
    func = functools.partial(
        extract_walk_csv,
        n_skip_edge_step=2,
        thd_n_step_each_walk=10,
        output_dir=args.output_dir
    )

    # Extract walks
    # Parallel map call - it blocks until the result is ready.
    pool.map(func, csv_files)


if __name__ == "__main__":
    main()