import numpy as np

from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.models import LinearAxis, Range1d
from bokeh.plotting import figure


def plot_acceleration(accs, output_html, max_plots=8):
    """Plot a list of Acceleration objects.
    
    from gaitcalibrate.util.plot_html import plot_acceleration
    import ntpath
    filename = ntpath.basename(filepath)
    filename = filename.split(".")[0]
    plot_acceleration(accs=[adj_w], output_html="debugs/{}.html".format(filename))
    
    """

    figures = []
    subplots = []
    subplot_idx = 0
    for i in range(len(accs[:max_plots])):
        acc = accs[i].data

        output_file(output_html)

        # Acceleration plot
        fig = figure(
            x_axis_type="datetime", 
            title="Acceleration {}".format(i+1), 
            plot_height=400,
            tools="hover,crosshair,pan,wheel_zoom,box_zoom,reset,",
        )
        fig.grid.grid_line_alpha=0.5
        fig.xaxis.axis_label = 'Time'
        fig.yaxis.axis_label = 'Acceleration (g)'
        fig.line(acc['dt'], acc['ver'], color='red', legend='ver (g)')
        fig.line(acc['dt'], acc['hor'], color='green', legend='hor (g)')
        fig.line(acc['dt'], acc['fwd'], color='blue', legend='fwd (g)')
        fig.legend.location = "top_right"

        # Add to the list of figures
        if subplot_idx == 0:
            subplots.append(fig)
            subplot_idx += 1
        else:
            subplots.append(fig)
            figures.append(subplots)
            subplots = []
            subplot_idx = 0

    if len(subplots) > 0:
        figures.append(subplots)

    # Grid plot
    all_figs = gridplot(
        figures,
        nbols=2,
        sizing_mode="scale_width"
    )

    # Show the results
    show(all_figs)


def plot_fc_detect(
    acc_dt,
    fwd_acc,
    filtered_fwd_acc, 
    peak_zc_down, 
    zc_ptom, 
    peak,
    output_html
):

    output_file(output_html)

    # Acceleration plot
    fig = figure(
        x_axis_type="datetime", 
        title="Foot contact detection", 
        plot_height=400,
        tools="hover,crosshair,pan,wheel_zoom,box_zoom,reset,",
    )
    fig.grid.grid_line_alpha=0.5
    fig.xaxis.axis_label = 'Time'
    fig.yaxis.axis_label = 'Acceleration (g) and Foot contact'
    fig.line(acc_dt, fwd_acc, color='blue', legend='raw (g)', alpha=0.3)
    fig.line(acc_dt, filtered_fwd_acc, color='blue', legend='filter (g)')
    # fig.circle(peak_zc_down, filtered_fwd_acc[peak_zc_down], color="red", legend="spd (m/s)", y_range_name="spd_dt", alpha=0.5)
    # fig.circle(zc_ptom, np.zeros(len(peak_zc_down)), color="black", legend="spd (m/s)", y_range_name="spd_dt", alpha=0.5)
    # fig.circle(peak, filtered_fwd_acc[peak], color="green", legend="spd (m/s)", y_range_name="spd_dt", alpha=0.5)
    fig.legend.location = "top_right"

    show(fig)
