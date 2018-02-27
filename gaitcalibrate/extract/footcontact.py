import numpy as np

from gaitcalibrate.util.helper import (get_zerocross_ptom, get_peak)


class FootContactPeakZCPtoM(object):

    def __init__(self, min_peak=0.05):
        self.min_peak = min_peak

    def get_footcontact(self, x):
        zc_ptom = get_zerocross_ptom(x)
        peak = get_peak(x, min_peak=self.min_peak)

        peak_zc_ptom = np.empty(0, dtype=int)
        for zc_idx, zc in enumerate(zc_ptom):
            peak_before_zc = peak[peak < zc]
            if zc_idx > 0:
                peak_before_zc = peak_before_zc[peak_before_zc > zc_ptom[zc_idx-1]]
            if len(peak_before_zc) > 0:
                # Use the max peak value
                max_idx = np.argmax(x[peak_before_zc])
                if x[peak_before_zc][max_idx] >= self.min_peak:
                    peak_zc_ptom = np.append(peak_zc_ptom, peak_before_zc[max_idx])

        return peak_zc_ptom, zc_ptom, peak
