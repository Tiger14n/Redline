"""Pure-numpy replacements for the two scipy.signal functions the solver uses:
find_peaks(x, height, distance) and medfilt(x, kernel_size).

These reproduce scipy's algorithms exactly (verified bit-for-bit in research/test_npsignal.py),
so the solver is scipy-free with zero behaviour change. Dropping scipy removes ~50 MB from
a bundled build.
"""
from __future__ import annotations
import numpy as np


def _local_maxima_1d(x):
    """scipy.signal._peak_finding_utils._local_maxima_1d: strict local maxima, plateaus
    reported at their (floored) midpoint, edges never maxima."""
    midpoints = []
    i, i_max = 1, x.shape[0] - 1
    while i < i_max:
        if x[i - 1] < x[i]:
            i_ahead = i + 1
            while i_ahead < i_max and x[i_ahead] == x[i]:
                i_ahead += 1
            if x[i_ahead] < x[i]:
                left, right = i, i_ahead - 1
                midpoints.append((left + right) // 2)
                i = i_ahead
        i += 1
    return np.asarray(midpoints, dtype=np.intp)


def _select_by_peak_distance(peaks, priority, distance):
    """scipy's greedy distance filter: keep highest-priority peaks, suppress any within
    ceil(distance) samples. Matches scipy._select_by_peak_distance."""
    n = peaks.shape[0]
    keep = np.ones(n, dtype=bool)
    distance_ = int(np.ceil(distance))
    order = np.argsort(priority)                    # ascending; iterate high->low
    for i in range(n - 1, -1, -1):
        j = order[i]
        if not keep[j]:
            continue
        k = j - 1
        while k >= 0 and peaks[j] - peaks[k] < distance_:
            keep[k] = False
            k -= 1
        k = j + 1
        while k < n and peaks[k] - peaks[j] < distance_:
            keep[k] = False
            k += 1
    return keep


def find_peaks(x, height=None, distance=None):
    """Subset of scipy.signal.find_peaks (height + distance only), same return shape
    (peaks, properties_dict). Order matches scipy: local maxima -> height -> distance."""
    x = np.asarray(x, dtype=np.float64)
    peaks = _local_maxima_1d(x)
    if height is not None and peaks.shape[0]:
        peaks = peaks[x[peaks] >= height]
    if distance is not None and peaks.shape[0]:
        peaks = peaks[_select_by_peak_distance(peaks, x[peaks], distance)]
    return peaks, {}


def medfilt(volume, kernel_size):
    """scipy.signal.medfilt for 1-D input: zero-padded sliding-window median, odd kernel."""
    x = np.asarray(volume, dtype=np.float64)
    n = x.shape[0]
    k = int(kernel_size) | 1                        # scipy requires odd
    if k <= 1 or n == 0:
        return x.copy()
    r = k // 2
    xp = np.concatenate([np.zeros(r), x, np.zeros(r)])
    idx = np.arange(n)[:, None] + np.arange(k)[None, :]
    return np.median(xp[idx], axis=1)
