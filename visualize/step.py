import argparse
import ntpath
import os

import numpy as np
import pandas as pd

from bokeh.io import output_file, show
from bokeh.layouts import column
from bokeh.plotting import figure

from gaitcalibrate.data.timeseries import Acceleration
from gaitcalibrate.extract.footcontact import FootContactPeakZCPtoM
from gaitcalibrate.extract.walk import extract_walk_csv
from gaitcalibrate.util.adjust_acceleration import tilt_adjustment, noise_filtering
from gaitcalibrate.util.helper import gen_idx_range


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acc_file", type=str, required=True,
                       help="File path to the CSV or NPY file that contains walking data.")
    parser.add_argument("--output_dir", type=str, default="outputs/step_plots",
                       help="Directory where to save outputs.")
    parser.add_argument("--sampling_rate", type=float, default=100.0,
                       help="Sampling rate of the acceleration file.")
    parser.add_argument("--body_location", type=str, default="lower_back",
                       help="Location of the device on human body.")
    parser.add_argument("--position", type=str, default="center_right",
                       help="Position of the device (e.g., center_right).")
    parser.add_argument("--preprocess", type=str, default=None,
                       help="Preprocessing algorithms (split by non-spacing commas) before step extraction (e.g., transform_orientation,tilt_adjustment).")
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
            thd_n_step_each_walk=4,
            sampling_rate=args.sampling_rate,
            body_location=args.body_location,
            position=args.position
        )
    # NPY file - contain a list of walks
    elif args.acc_file.lower().endswith('.npy'):
        walks = np.load(args.acc_file)
    else:
        raise Exception("Invalid acc_file.")

    # Html file
    output_file(os.path.join(args.output_dir, "{}.html".format(filename)))

    # Using Bokeh to create html plots
    figs = []
    for i in range(len(walks)):
        # Tilt Adjustment and Noise Filtering
        acc = walks[i]
        acc = tilt_adjustment(acc=acc)
        filtered_fwd_acc = noise_filtering(
            acc=acc, 
            lowpass_f=0.1,
            highpass_f=6,
            mode="bandpass"
        ).data['fwd'].values

        acc = noise_filtering(
            acc=acc, 
            lowpass_f=20,
            mode="lowpass"
        )

        # Get foot-contact for walking speed estimation
        fc_detector = FootContactPeakZCPtoM(min_peak=0.01)
        peak_zc_down, zc_ptom, peak = fc_detector.get_footcontact(x=filtered_fwd_acc)
        idx_fc = peak_zc_down

        # # Step extraction
        # idx_sample = gen_idx_range(idx_fc, len(acc.data))
        # steps = np.empty(len(idx_sample), dtype=object)
        # for s_idx in xrange(len(steps)):
        #     idx_step = idx_sample[s_idx]
        #     steps[s_idx] = acc.get_idx(idx_step)

        # Get only DataFrame of the acceleration data
        acc = acc.data

        # Plot foot contact detection
        fig = figure(
            x_axis_type="datetime", 
            title="Acceleration", 
            plot_width=1000,
            plot_height=600,
            tools="hover,crosshair,pan,wheel_zoom,box_zoom,undo,redo,reset,"
        )
        fig.grid.grid_line_alpha=0.5
        fig.xaxis.axis_label = 'Time'
        fig.yaxis.axis_label = 'Acceleration (g)'

        # Acceleration
        fig.line(acc['dt'], acc['ver'], color='red', legend='ver', alpha=0.2)
        fig.line(acc['dt'], acc['hor'], color='green', legend='hor', alpha=0.2)
        fig.line(acc['dt'], acc['fwd'], color='blue', legend='fwd', alpha=0.2)
        # Foot-contact
        fig.line(acc['dt'], filtered_fwd_acc, color='blue', legend='fwd')
        fig.scatter(acc.iloc[peak_zc_down]['dt'], filtered_fwd_acc[peak_zc_down], marker="circle", size=12, line_color="red", fill_color="navy", legend="peak_zc_down")
        fig.scatter(acc.iloc[peak]['dt'], filtered_fwd_acc[peak], marker="circle", size=8, line_color="black", fill_color="blue", legend="peak")
        fig.scatter(acc.iloc[zc_ptom]['dt'], np.zeros(len(zc_ptom)), marker="circle", size=8, line_color="black", fill_color="black", legend="zc_ptom")
        fig.legend.location = "top_right"

        figs.append(fig)

    # Show the results
    p = column(
        figs
    )
    show(p)


if __name__ == "__main__":
    main()

# # Display - step
# n_plot_steps = min(16, len(steps))
# rand_idx = random.sample(range(n_plot_steps), n_plot_steps)
# plt.figure(figsize=(fig_width, fig_height), dpi=fig_save_dpi)
# plt.suptitle('Examples of steps', fontsize=20)
# for s_idx in xrange(n_plot_steps):
#     plt.subplot(4, 4, s_idx + 1)
#     step_idx = rand_idx[s_idx]
#     plt.plot(steps[step_idx].data['ver'], color='r', label='ver')
#     plt.plot(steps[step_idx].data['hor'], color='g', label='hor')
#     plt.plot(steps[step_idx].data['fwd'], color='b', label='fwd')
#     plt.ylabel('Acceleration ($m/s^{2}$)', fontsize=18)
#     plt.xlabel('Sample', fontsize=18)
#     ax = plt.gca()
#     ax.text(0.95, 0.9, '{} m/s'.format(step_speeds[step_idx]),
#             horizontalalignment='right',
#             verticalalignment='top',
#             transform=ax.transAxes,
#             bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 5})
# plt.tight_layout()
# plt.savefig(os.path.join(app.config['TMP_FOLDER'], 'steps_{}.png'.format(f_idx)))