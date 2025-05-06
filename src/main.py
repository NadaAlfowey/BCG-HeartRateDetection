# Import required libraries
import math
import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, resample
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from band_pass_filtering import band_pass_filtering
from compute_vitals import vitals
from detect_apnea_events import apnea_events
from detect_body_movements import detect_patterns
from modwt_matlab_fft import modwt
from modwt_mra_matlab_fft import modwtmra
from remove_nonLinear_trend import remove_nonLinear_trend
from data_subplot import data_subplot
from detect_peaks import detect_peaks

# ======================================================================================================================

print('\nstart processing ...')

file = '../data/BCG/fixed_bcg.csv'
reference_rr_file = '../data/BCG/Reference/RR/02_20231103_RR.csv'

if file.endswith(".csv"):
    fileName = os.path.join(file)
    if os.stat(fileName).st_size != 0:
        rawData = pd.read_csv(fileName)
        utc_time = rawData["Timestamp"].values
        print("\U0001F4C5 BCG Start:", pd.to_datetime(utc_time.min(), unit='ms'))
        print("\U0001F4C5 BCG End  :", pd.to_datetime(utc_time.max(), unit='ms'))
        data_stream = rawData["BCG"].values
        fs = int(rawData["fs"].iloc[0])
        print("✅ Loaded samples:", len(data_stream))
        print("✅ Sampling frequency (Hz):", fs)

        # Load ECG-derived reference HR
        rr_data = pd.read_csv(reference_rr_file)
        rr_data['Timestamp'] = pd.to_datetime(rr_data['Timestamp'])

        print("\U0001F4C5 RR Start :", rr_data['Timestamp'].min())
        print("\U0001F4C5 RR End   :", rr_data['Timestamp'].max())

        rr_data['Timestamp'] = rr_data['Timestamp'].astype(np.int64) // 10**6

        # Synchronize time range
        start_time = max(utc_time.min(), rr_data['Timestamp'].min())
        end_time = min(utc_time.max(), rr_data['Timestamp'].max())
        mask_bcg = (utc_time >= start_time) & (utc_time <= end_time)
        mask_rr = (rr_data['Timestamp'] >= start_time) & (rr_data['Timestamp'] <= end_time)

        synced_bcg = data_stream[mask_bcg]
        synced_utc = utc_time[mask_bcg]
        synced_rr = rr_data.loc[mask_rr]
        reference_hr = synced_rr['Heart Rate'].replace(0, np.nan).dropna().to_numpy()

        print("\U0001F552 Sync window (ms):", start_time, "→", end_time)
        print("\U0001F4CA Synced BCG samples:", len(synced_bcg))
        print("\U0001F4CA Synced RR samples:", len(reference_hr))

        # Denoise + Remove Movement Artifacts
        start_point, end_point, window_shift = 0, 500, 500
        clean_bcg, clean_time = detect_patterns(start_point, end_point, window_shift, synced_bcg, synced_utc, plot=1)

        # Filter for HR and respiratory components
        movement = band_pass_filtering(clean_bcg, fs, "bcg")
        breathing = band_pass_filtering(clean_bcg, fs, "breath")
        breathing = remove_nonLinear_trend(breathing, 3)
        breathing = savgol_filter(breathing, 11, 3)

        # Wavelet transform
        w = modwt(movement, 'bior3.9', 4)
        dc = modwtmra(w, 'bior3.9')
        wavelet_cycle = dc[4]

        # Peak detection from filtered BCG
        peaks = detect_peaks(wavelet_cycle, mpd=fs//2)
        peak_times = clean_time[peaks] / 1000.0  # ms to seconds
        rr_intervals_sec = np.diff(peak_times)
        estimated_hr = 60.0 / rr_intervals_sec

        # Align arrays
        min_len = min(len(estimated_hr), len(reference_hr))
        estimated_hr = estimated_hr[:min_len]
        reference_hr = reference_hr[:min_len]

        # Metrics
        mae = mean_absolute_error(reference_hr, estimated_hr)
        rmse = np.sqrt(mean_squared_error(reference_hr, estimated_hr))
        mape = np.mean(np.abs((reference_hr - estimated_hr) / reference_hr)) * 100

        print('\nHeart Rate Comparison Metrics')
        print('MAE:', mae)
        print('RMSE:', rmse)
        print('MAPE:', mape)

        # Bland-Altman
        mean_hr = (estimated_hr + reference_hr) / 2
        diff_hr = estimated_hr - reference_hr
        plt.figure()
        plt.scatter(mean_hr, diff_hr)
        plt.axhline(np.mean(diff_hr), color='gray')
        plt.axhline(np.mean(diff_hr) + 1.96*np.std(diff_hr), color='red', linestyle='--')
        plt.axhline(np.mean(diff_hr) - 1.96*np.std(diff_hr), color='red', linestyle='--')
        plt.title("Bland-Altman Plot")
        plt.xlabel("Mean HR")
        plt.ylabel("Difference (Estimated - Reference)")
        plt.savefig('../results/bland_altman.png')

        # Pearson
        r, _ = pearsonr(reference_hr, estimated_hr)
        plt.figure()
        plt.scatter(reference_hr, estimated_hr)
        plt.title(f"Pearson Correlation (r={r:.2f})")
        plt.xlabel("Reference HR")
        plt.ylabel("Estimated HR")
        plt.savefig('../results/pearson_correlation.png')

        # Vitals
        t1, t2, window_length, window_shift = 0, 500, 500, 500
        limit = int(math.floor(breathing.size / window_shift))
        hr_beats = vitals(t1, t2, window_shift, limit, wavelet_cycle, clean_time, mpd=1, plot=0)
        print('\nHeart Rate Information')
        print('Minimum pulse : ', np.around(np.min(hr_beats)))
        print('Maximum pulse : ', np.around(np.max(hr_beats)))
        print('Average pulse : ', np.around(np.mean(hr_beats)))

        breath_beats = vitals(t1, t2, window_shift, limit, breathing, clean_time, mpd=1, plot=0)
        print('\nRespiratory Rate Information')
        print('Minimum breathing : ', np.around(np.min(breath_beats)))
        print('Maximum breathing : ', np.around(np.max(breath_beats)))
        print('Average breathing : ', np.around(np.mean(breath_beats)))

        events = apnea_events(breathing, clean_time, thresh=0.3)

        # Visualization
        t1, t2 = 2500, 5000
        data_subplot(clean_bcg, movement, breathing, wavelet_cycle, t1, t2)

    print('\nEnd processing ...')
