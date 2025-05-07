"""
Created on %(25/09/2017)
Function to compute vitals, i.e., heart rate and respiration.
"""

import numpy as np
import heartpy

from beat_to_beat import compute_rate


def vitals(t1, t2, win_size, window_limit, sig, time, mpd, plot=0, sig_type = "bcg"):
    all_rate = []
    for j in range(0, window_limit):
        sub_signal = sig[t1:t2]
        if sig_type == "ecg":
            w, results = heartpy.process(sub_signal, 50)
            rate = results['bpm']
            all_rate.append(rate)
        else:
            [rate, indices] = compute_rate(sub_signal, time, mpd)
            all_rate.append(rate)
        t1 = t2
        t2 += win_size
    all_rate = np.vstack(all_rate).flatten()
    return all_rate
