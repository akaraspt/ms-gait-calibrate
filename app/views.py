import os
import glob
import pickle
import simplejson
import time

import pandas as pd
import numpy as np
from bokeh.embed import components
from bokeh.layouts import gridplot
from bokeh.models import LinearAxis, Range1d
from bokeh.plotting import figure
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8
from flask import (Flask, render_template, request, 
                   redirect, url_for, send_from_directory)
from werkzeug.utils import secure_filename

from gaitcalibrate import dt_format
from gaitcalibrate.train import calibrate_model

from app import app
from estimate import estimate_speed
from prepare import create_walking_graphs, generate_train_data
from upload_file import IGNORED_FILES, gen_file_name, allowed_file, uploadfile


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/estimate', methods=['GET', 'POST'])
def estimate():
    models = []
    metas = []
    for f in os.listdir(app.config['MODEL_FOLDER']):
        if os.path.isfile(os.path.join(app.config['MODEL_FOLDER'], f)) and f not in IGNORED_FILES:
            models.append(f)
            with open(os.path.join(app.config['MODEL_FOLDER'], f), 'rb') as ff:
                model = pickle.load(ff)
                sampling_rate = str(model['sampling_rate'])
                body_loc = model['body_location']
                pos = model['position']
                metas.append(','.join([sampling_rate, body_loc, pos]))
    files = [ f for f in os.listdir(app.config['UPLOAD_FOLDER']) if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'],f)) and f not in IGNORED_FILES ]
    submit_btn = request.form.get('submit_btn')

    if request.method == 'GET':
        param_dict = {
            "models": zip(models, metas),
            "files": files
        }
        return render_template("estimate.html", param_dict=param_dict)
    if request.method == 'POST':
        selected_file = request.form.get('selected_file')
        selected_model = request.form.get('selected_model')

        est_output = estimate_speed(selected_file=selected_file, selected_model=selected_model)

        # No walk detected
        if len(est_output["walks"]) == 0:
            raise Exception("Not yet implemented.")
            # param_dict = {
            #     "models": zip(models, metas),
            #     "files": files,
            #     "selected_file": selected_file,
            #     "selected_model": selected_model
            # }
            # param_dict.update(output_walk)
            # return render_template("estimate.html", param_dict=param_dict)

        # Create output figures with Bokeh
        figures = []
        subplots = []
        subplot_idx = 0
        for i in range(len(est_output["walks"])):
            acc = est_output["walks"][i].data
            out = est_output["outputs"][i]
            if out == -1:
                est_dt = acc['dt']
                est_spd = np.zeros(len(acc["dt"].values))
                walk_spd = 0.0
            else:
                est_dt = out["step_dt"]
                est_spd = out["step_spd"]
                walk_spd = out["walk_spd"]

            # Acceleration plot
            fig = figure(
                x_axis_type="datetime", 
                title="Walk {} (estimated speed={:.2f} m/s)".format(i+1, walk_spd), 
                plot_height=400,
                tools="hover,crosshair,pan,wheel_zoom,reset,",
            )
            fig.grid.grid_line_alpha=0.5
            fig.xaxis.axis_label = 'Time'
            fig.yaxis.axis_label = 'Acceleration (g)'
            fig.line(acc['dt'], acc['ver'], color='red', legend='ver (g)')
            fig.line(acc['dt'], acc['hor'], color='green', legend='hor (g)')
            fig.line(acc['dt'], acc['fwd'], color='blue', legend='fwd (g)')
            fig.legend.location = "top_right"

            # Adding the estimated speed axis to the plot
            fig.extra_y_ranges = {"spd_dt": Range1d(start=0, end=2)}
            fig.add_layout(
                LinearAxis(y_range_name="spd_dt", axis_label="Estimated speed (m/s)"),
                'right'
            )
            # Estimated walking speed plot
            fig.circle(est_dt, est_spd, color="black", legend="spd (m/s)", y_range_name="spd_dt")
            fig.line(est_dt, est_spd, color='black', legend='spd (m/s)', alpha=0.5, y_range_name="spd_dt")

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

        js_resources = INLINE.render_js()
        css_resources = INLINE.render_css()

        plot_script, plot_div = components(all_figs)

        param_dict = {
            "models": zip(models, metas),
            "files": files,
            "selected_file": selected_file,
            "selected_model": selected_model,
            "js_resources": js_resources,
            "css_resources": css_resources,
            "plot_div": plot_div,
            "plot_script": plot_script,
            "status": "success"
        }
        return render_template("estimate.html", param_dict=param_dict)


@app.route('/calibrate', methods=['GET', 'POST'])
def calibrate():
    if request.method == 'GET':
        return render_template("calibrate.html")
    if request.method == 'POST':
        train_files = []
        train_metas = []
        root = app.config['CALIBRATE_FOLDER']
        for path, subdirs, files in os.walk(root):
            for name in files:
                if "meta_" in name and name.lower().endswith('.csv'):
                    train_files.append(os.path.join(path, name))
                    with open(os.path.join(path, name), 'r') as ff:
                        # 100,lower_back,center_right,0.86
                        train_metas.append(ff.readline())

        submit_btn = request.form.get('submit_btn')

        if submit_btn == 'Check':
            param_dict = {
                "files": zip(train_files, train_metas)
            }
            return render_template("calibrate.html", param_dict=param_dict)

        elif submit_btn == 'Calibrate':
            selected_files = request.form.getlist('selected_files[]')
            model_name = request.form.get('model_name')

            # Calibrate a model
            model = calibrate_model(
                train_files=selected_files,
                n_skip_edge_step=2,
                thd_n_step_each_walk=3
            )

            # Save calibrated model
            print 'Save the calibrated model ...'
            filepath = os.path.join(app.config['MODEL_FOLDER'], model_name + '.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)

            param_dict = {
                "files": zip(train_files, train_metas),
                "model_name": model_name
            }
            return render_template("calibrate.html", param_dict=param_dict)


@app.route('/upload')
def upload():
    return render_template("upload.html")


@app.route('/prepare', methods=['GET', 'POST'])
def prepare():
    if request.method == 'GET':
        files = [ f for f in os.listdir(app.config['UPLOAD_FOLDER']) if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'],f)) and f not in IGNORED_FILES ]
        files.sort()
        selected_file = request.form.get('selected_file')

        param_dict = {
            "files": files,
            "selected_file": selected_file
        }
        return render_template("prepare.html", param_dict=param_dict)

    if request.method == 'POST':
        files = [ f for f in os.listdir(app.config['UPLOAD_FOLDER']) if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'],f)) and f not in IGNORED_FILES ]
        files.sort()
        selected_file = request.form.get('selected_file')
        sampling_rate = int(request.form.get('sampling_rate'))
        body_loc = request.form.get('body_loc')
        pos = request.form.get('pos')
        submit_btn = request.form.get('submit_btn')

        if submit_btn == 'Visualize':
            figpath = create_walking_graphs(
                filepath=os.path.join(app.config['UPLOAD_FOLDER'], selected_file),
                sampling_rate=sampling_rate,
                body_loc=body_loc,
                pos=pos
            )

            param_dict = {
                "files": files,
                "selected_file": selected_file,
                "sampling_rate": sampling_rate,
                "body_loc": body_loc,
                "pos": pos,
                "selected_walk_figpath": figpath,
                "timestamp": time.time()
            }
            return render_template("prepare.html", param_dict=param_dict)
        elif submit_btn == 'Create':
            walk_idx = [int(i) for i in request.form.getlist('walkidx[]')]
            walk_speed = [float(s) for s in request.form.getlist('walkspeed[]')]
            assert len(walk_idx) == len(walk_speed)
            walk_idx_speed = [(walk_idx[i], walk_speed[i]) for i in range(len(walk_idx))]
            selected_walk_figpath = request.form.get('selected_walk_figpath')

            figpath = generate_train_data(
                filepath=os.path.join(app.config['UPLOAD_FOLDER'], selected_file),
                sampling_rate=sampling_rate,
                body_loc=body_loc,
                pos=pos,
                selected_walk_idx=walk_idx,
                selected_walk_spd=walk_speed
            )

            param_dict = {
                "files": files,
                "selected_file": selected_file,
                "sampling_rate": sampling_rate,
                "body_loc": body_loc,
                "pos": pos,
                "selected_walk_figpath": selected_walk_figpath,
                "selected_speed_figpath": figpath,
                "walk_idx_speed": walk_idx_speed,
                "timestamp": time.time()
            }
            return render_template("prepare.html", param_dict=param_dict)


@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['files[]']
        #pprint (vars(objectvalue))

        if file:
            filename = secure_filename(file.filename)
            filename = gen_file_name(filename)
            mimetype = file.content_type


            if not allowed_file(file.filename):
                result = uploadfile(name=filename, type=mimetype, size=0, not_allowed_msg="Filetype not allowed")

            else:
                # save file to disk
                uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(uploaded_file_path)

                # create thumbnail after saving
                if mimetype.startswith('image'):
                    create_thumbnai(filename)
                
                # get file size after saving
                size = os.path.getsize(uploaded_file_path)

                # return json for js call back
                result = uploadfile(name=filename, type=mimetype, size=size)
            
            return simplejson.dumps({"files": [result.get_file()]})

    if request.method == 'GET':
        # get all file in ./data directory
        files = [ f for f in os.listdir(app.config['UPLOAD_FOLDER']) if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'],f)) and f not in IGNORED_FILES ]
        
        file_display = []

        for f in files:
            size = os.path.getsize(os.path.join(app.config['UPLOAD_FOLDER'], f))
            file_saved = uploadfile(name=f, size=size)
            file_display.append(file_saved.get_file())

        return simplejson.dumps({"files": file_display})

    return redirect(url_for('upload'))


@app.route("/data/<path:filename>", methods=['GET'])
def get_file(filename):
    absolute_upload_path = os.path.join("..", app.config['UPLOAD_FOLDER'])
    return send_from_directory(absolute_upload_path, filename=filename)


@app.route("/delete/<path:filename>", methods=['DELETE'])
def delete(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file_thumb_path = os.path.join(app.config['THUMBNAIL_FOLDER'], filename)

    if os.path.exists(file_path):
        try:
            # Remove file
            os.remove(file_path)

            # Remove thumbnail if available
            if os.path.exists(file_thumb_path):
                os.remove(file_thumb_path)
            
            return simplejson.dumps({filename: 'True'})
        except:
            return simplejson.dumps({filename: 'False'})


@app.route("/visualize", methods=['GET', 'POST'])
def visualize():
    if request.method == 'GET':
        files = [ f for f in os.listdir(app.config['UPLOAD_FOLDER']) if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'],f)) and f not in IGNORED_FILES ]
        selected_file = request.form.get('selected_file')

        param_dict = {
            "files": files,
            "selected_file": selected_file
        }
        return render_template("visualize.html", param_dict=param_dict)
    elif request.method == 'POST':
        files = [ f for f in os.listdir(app.config['UPLOAD_FOLDER']) if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'],f)) and f not in IGNORED_FILES ]
        selected_file = request.form.get('selected_file')
        submit_btn = request.form.get('submit_btn')

        if submit_btn == 'Visualize':
            filepath=os.path.join(app.config['UPLOAD_FOLDER'], selected_file)

            # Read the first two line to determine how to load CSV
            with open(filepath) as f:
                head = [next(f) for x in xrange(2)]

            if len(head[-1].split(",")) == 4:
                column_names = ['dt', 'x', 'y', 'z']
            elif len(head[-1].split(",")) == 5:
                column_names = ['dt', 'x', 'y', 'z', 'spd']
            else:
                raise Exception("Invalid CSV file.")

            def dateparse(x): return pd.datetime.strptime(x, dt_format)
            
            if "lower_back" in head[0]:
                acc = pd.read_csv(filepath,
                                names=column_names,
                                header=None,
                                parse_dates=['dt'],
                                skiprows=1,
                                date_parser=dateparse)
            else:
                acc = pd.read_csv(filepath,
                                names=column_names,
                                header=None,
                                parse_dates=['dt'],
                                date_parser=dateparse)

            fig = figure(
                x_axis_type="datetime", 
                title="Acceleration", 
                sizing_mode="scale_width",
                plot_height=400,
                tools="hover,crosshair,pan,wheel_zoom,box_zoom,undo,redo,reset,"
            )
            fig.grid.grid_line_alpha=0.5
            fig.xaxis.axis_label = 'Time'
            fig.yaxis.axis_label = 'Acceleration (g)'

            fig.line(acc['dt'], acc['x'], color='red', legend='x')
            fig.line(acc['dt'], acc['y'], color='green', legend='y')
            fig.line(acc['dt'], acc['z'], color='blue', legend='z')
            fig.legend.location = "top_right"

            js_resources = INLINE.render_js()
            css_resources = INLINE.render_css()

            plot_script, plot_div = components(fig)

            param_dict = {
                "files": files,
                "selected_file": selected_file,
                "js_resources": js_resources,
                "css_resources": css_resources,
                "plot_div": plot_div,
                "plot_script": plot_script
            }
            html = render_template("visualize.html", param_dict=param_dict)
            return encode_utf8(html)
