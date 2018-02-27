import pickle
import os
import re

import numpy as np
import pandas as pd

from numpy.random import RandomState
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

from gaitcalibrate import dt_format
from gaitcalibrate.data.timeseries import Acceleration, WalkSpeed
from gaitcalibrate.extract.step import extract_step
from gaitcalibrate.feature.step import StepFeatureExtractor


def build_svr(x, y, n_jobs=4):
    """Build a SVR model for walking speed estimation."""

    # Randomness
    random_seed = 1234567890
    prng = RandomState(random_seed)

    # Feature extraction
    feature_ext = StepFeatureExtractor()
    x = feature_ext.extract(x)

    # Feature scaling
    scaler = preprocessing.StandardScaler()
    scaler = scaler.fit(x)
    scaled_x = scaler.transform(X=x)

    # Split the dataset in training and test sets
    x_train, x_test, y_train, y_test = train_test_split(scaled_x, y,
                                                        test_size=0.1,
                                                        random_state=prng)

    print 'Grid search via Cross-validation'
    print 'Data set ... X:', scaled_x.shape, 'Y:', y.shape
    print 'Train set ... X:', x_train.shape, 'Y:', y_train.shape
    print 'Test set ... X:', x_test.shape, 'Y:', y_test.shape
    print ' '
    print 'Train an estimator ...'

    # Set the parameters by cross-validation
    C_range = 2.0 ** np.arange(-8, 9, 2)
    gamma_range = 2.0 ** np.arange(-8, 9, 2)
    epsilon_range = np.asarray([0.01])
    tuned_parameters = [
        {'kernel': ['rbf'], 'gamma': gamma_range, 'C': C_range, 'epsilon': epsilon_range},
        {'kernel': ['linear'], 'C': C_range, 'epsilon': epsilon_range}
    ]

    # Cross-validation set
    kf = KFold(n_splits=10,
               shuffle=True,
               random_state=prng)
    cv = kf.get_n_splits(len(x_train))

    # Use mean squared error scoring method
    score = 'neg_mean_squared_error'

    print ''
    print '# Tuning hyper-parameters for %s' % score

    # Grid search with cross-validation
    grid_search_svr = GridSearchCV(
        SVR(C=1),
        param_grid=tuned_parameters,
        cv=cv,
        scoring=score,
        n_jobs=n_jobs,
        verbose=True
    )
    grid_search_svr.fit(x_train, y_train)

    print ''
    print 'Grid scores on development set:'

    means = grid_search_svr.cv_results_['mean_test_score']
    stds = grid_search_svr.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_search_svr.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    print ''
    print 'Best parameters set found on development set:'

    print grid_search_svr.best_estimator_

    estimator = grid_search_svr.best_estimator_
    grid_search_estimator = grid_search_svr

    print ''
    print '========================================='
    print ''

    print 'Performance on test set:'

    # Evaluate model based on test set
    y_true, y_pred = y_test, grid_search_estimator.predict(x_test)
    mse = mean_squared_error(y_true, y_pred)
    error = y_true - y_pred

    # Sort with error
    sort_idx = np.argsort(np.abs(error))
    print 'Mean squared error:', mse
    print 'Actual\t\tPredict\t\tError'
    for i in range(len(sort_idx)):
        print '{}\t\t{}\t\t{}'.format(y_true[sort_idx[i]], y_pred[sort_idx[i]], error[sort_idx[i]])

    model = {
        'estimator': estimator,
        'grid_search_estimator': grid_search_estimator,
        'scaler': scaler,
        'feature_ext': feature_ext
    }

    return model


def calibrate_model(
    train_files,
    fn_build_model=build_svr,
    n_skip_edge_step=2,
    thd_n_step_each_walk=3
):
    """Calibrate a model for walking speed estimation.
    
    Assuming that all of the training files have been preprocessed.
    Specifically, each file contains metadata string at the first line.
    The following lines are acceleration of one walk period that are 
    in the following format:

    - line 1: sampling_rate,body_location,position,label_spd
    - line 2: timestamp,ver,hor,fwd,spd
    - line 3: timestamp,ver,hor,fwd,spd
    - line 4: ...
    
    """

    # Steps and their walking speed labels
    all_steps = []
    all_speeds = []
    sampling_rate = None
    body_location = None
    position = None

    # Get all steps and their corresponding speed from all training files
    for f_idx, filepath in enumerate(train_files):
        ############
        # Metadata #
        ############
        # Read the first line to get the metadata
        with open(filepath, 'r') as f:
            first_line = f.readline()
            if re.search('[a-zA-Z]', first_line) is not None:
                cols = first_line.split(",")
                sampling_rate = float(cols[0].strip())
                body_location = cols[1].strip()
                position = cols[2].strip()
                label_spd = float(cols[3].strip())

                # Skip metadata line
                skiprows = 1
            else:
                raise Exception("Invalid training file.")

        print "{} (meta={}):fs={},body={},pos={},spd={}".format(
            filepath, 
            skiprows == 1,
            sampling_rate, 
            body_location, 
            position,
            label_spd
        )

        ################################################
        # Create Acceleration and WalkingSpeed objects #
        ################################################
        # Function to parse datetime format
        def dateparse(x): return pd.datetime.strptime(x, dt_format)
        acc_spd = pd.read_csv(
            filepath,
            names=['dt', 'ver', 'hor', 'fwd', 'spd'],
            header=None,
            parse_dates=['dt'],
            date_parser=dateparse,
            skiprows=skiprows
        )

        # Sanity check
        assert abs(np.mean(acc_spd['spd'].values) - label_spd) < 1e-10

        # Get acceleration data of a walk and its walking speed
        adj_w = Acceleration(data=acc_spd[acc_spd.columns[:-1]],
                             fs=sampling_rate)
        spd = WalkSpeed(data=acc_spd[[acc_spd.columns[0], acc_spd.columns[-1]]])

        ###################
        # Step extraction #
        ###################
        steps = extract_step(
            acc=adj_w,
            g2acc=True
        )

        # Remove edge steps which might not be stable
        if n_skip_edge_step > 0:
            steps = steps[n_skip_edge_step:-n_skip_edge_step]

        # Check whether the number of steps are sufficient to do walking speed estimation
        if len(steps) > thd_n_step_each_walk:
            # Append steps and their walking speed labels of this walk
            all_steps.append(steps)
            step_speeds = np.empty(len(steps), dtype=float)
            for s_idx in xrange(len(step_speeds)):
                step_speeds[s_idx] = np.mean(
                    spd.data.iloc[steps[s_idx].data.index.values]['spd'].values
                )
            all_speeds.append(step_speeds)
        else:
            print 'Skip: insufficient steps after removing the edge steps'

    # Stack steps and speeds from all training files
    all_steps = np.hstack(all_steps)
    all_speeds = np.hstack(all_speeds)

    # Build model for walking speed estimation
    model = fn_build_model(
        x=all_steps,
        y=all_speeds
    )
    model.update({
        'sampling_rate': sampling_rate,
        'body_location': body_location,
        'position': position
    })

    return model
