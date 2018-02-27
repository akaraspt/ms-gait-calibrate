import numpy as np
import pandas as pd


from gaitcalibrate import dt_format
from gaitcalibrate.data.timeseries import Acceleration, WalkSpeed
from gaitcalibrate.util.adjust_acceleration import transform_orientation


class CSVAccelerationReader(object):

    def __init__(self):
        pass

    def load(self,
             filepath,
             body_loc,
             pos,
             fs):
        def dateparse(x): return pd.datetime.strptime(x, dt_format)
        acc = pd.read_csv(filepath,
                          names=['dt', 'x', 'y', 'z'],
                          header=None,
                          parse_dates=['dt'],
                          date_parser=dateparse)

        # 3-D acceleration will be set according to the orientation of the device attached
        # to the human body
        transformed_acc = transform_orientation(data=acc,
                                                body_loc=body_loc,
                                                pos=pos)

        acc = Acceleration(data=transformed_acc,
                           fs=fs)

        return acc


class CSVSpeedReader(object):

    def __init__(self):
        pass

    def load(self,
             filepath):
        def dateparse(x): return pd.datetime.strptime(x, dt_format)
        spd = pd.read_csv(filepath,
                          names=['dt', 'spd'],
                          header=None,
                          parse_dates=['dt'],
                          date_parser=dateparse)

        spd = WalkSpeed(data=spd)

        return spd


class CSVAccelerationSpeedReader(object):

    def __init__(self):
        pass

    def load(self,
             filepath,
             fs):
        def dateparse(x): return pd.datetime.strptime(x, dt_format)
        acc_spd = pd.read_csv(filepath,
                              names=['dt', 'ver', 'hor', 'fwd', 'spd'],
                              header=None,
                              parse_dates=['dt'],
                              date_parser=dateparse)

        acc = Acceleration(data=acc_spd[acc_spd.columns[:-1]],
                           fs=fs)
        spd = WalkSpeed(data=acc_spd[[acc_spd.columns[0], acc_spd.columns[-1]]])

        return acc, spd
