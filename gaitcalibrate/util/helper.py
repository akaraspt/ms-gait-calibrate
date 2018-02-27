import numpy as np


def get_zerocross_ptom(x):
    diff_sign = np.diff(np.sign(x))
    idx_ptom = np.where(diff_sign < 0)[0]
    zc = np.where(diff_sign)[0]
    zc_ptom = np.intersect1d(zc, idx_ptom)
    return zc_ptom


def get_peak(x,
             min_peak=0.05):
    from scipy.signal import argrelextrema

    # Get peaks
    peak = argrelextrema(x, np.greater)[0]

    # Filter peak with min_peak
    peak = peak[x[peak] >= min_peak]

    return peak


def gen_idx_range(input_idx, end_idx):
    # Get around indexing problem of arange()
    idx = input_idx + 1

    # Create ndarray of index
    idx_range = np.empty(idx.size + 1, dtype=object)
    for i in range(idx_range.size):
        if i == 0:
            idx_range[i] = np.arange(idx[i])
        elif i == idx_range.size - 1:
            idx_range[i] = np.arange(idx[i - 1], end_idx)
        else:
            idx_range[i] = np.arange(idx[i - 1], idx[i])
    return idx_range


def get_list_seq(num):
    """Get a list of sequence integer numbers.

    Parameters
    ----------
    num : ``ndarray``
        Integer numbers sorted in ascending order.

    Returns
    -------
    list_seq : ``list``
        List of sequence numbers.

    """

    # Get indicies of the end index of each sequence (if any)
    if num is not None and len(num) > 0:
        end_seq_idx = np.where(np.diff(num) > 1)[0]
        end_seq_idx = np.append(end_seq_idx, [(len(num) - 1)])
        end_seq_idx += 1

        # Get a list of sequence
        list_seq = []
        for i, e in enumerate(end_seq_idx):
            if i == 0:
                list_seq.append(num[xrange(e)])
            else:
                list_seq.append(num[xrange(end_seq_idx[i-1], e)])
        return list_seq
    else:
        return []
