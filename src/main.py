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
# ======================================================================================================================

# # convert from epoch to datetime
# epoch_ms = 1699022112866
# dt = datetime.fromtimestamp(epoch_ms / 1000)
# print("Datetime:", dt)


# Main program starts here
print('\nstart processing ...')

file = '../data/BCG/fixed_bcg.csv'
reference_rr_file = '../data/BCG/Reference/RR/02_20231103_RR.csv'


if file.endswith(".csv"):
    fileName = os.path.join(file)
    if os.stat(fileName).st_size != 0:
        rawData = pd.read_csv(fileName) 
        utc_time = rawData["Timestamp"].values
        print("📅 BCG Start:", pd.to_datetime(utc_time.min(), unit='ms'))
        print("📅 BCG End  :", pd.to_datetime(utc_time.max(), unit='ms'))
        data_stream = rawData["BCG"].values
        fs = int(rawData["fs"].iloc[0])
        print("✅ Loaded samples:", len(data_stream))
        print("✅ Sampling frequency (Hz):", fs)

        # Load ECG-derived reference HR
        rr_data = pd.read_csv(reference_rr_file)
        rr_data['Timestamp'] = pd.to_datetime(rr_data['Timestamp'])
        reference_hr = rr_data['Heart Rate'].to_numpy()

        print("📅 RR Start :", rr_data['Timestamp'].min())
        print("📅 RR End   :", rr_data['Timestamp'].max())

        # Convert RR timestamps to UNIX epoch milliseconds
        rr_data['Timestamp'] = pd.to_datetime(rr_data['Timestamp'])
        rr_data['Timestamp'] = rr_data['Timestamp'].astype(np.int64) // 10**6  # convert to ms


        # Synchronize time range
        start_time = max(utc_time.min(), rr_data['Timestamp'].min())
        end_time = min(utc_time.max(), rr_data['Timestamp'].max())
        mask_bcg = (utc_time >= start_time) & (utc_time <= end_time)
        mask_rr = (rr_data['Timestamp'] >= start_time) & (rr_data['Timestamp'] <= end_time)

        synced_bcg = data_stream[mask_bcg]
        synced_rr = rr_data.loc[mask_rr]    

        print("🕒 Sync window (ms):", start_time, "→", end_time)
        print("📊 Synced BCG samples:", len(synced_bcg))
        print("📊 Synced RR samples:", len(synced_rr))

        data_stream = data_stream[mask_bcg]
        utc_time = utc_time[mask_bcg]
        rr_data = rr_data[mask_rr]
        reference_hr = rr_data['Heart Rate'].to_numpy()

        start_point, end_point, window_shift, fs = 0, 500, 500, 50
        # ==========================================================================================================
        print("Total BCG samples:", len(data_stream))
        data_stream, utc_time = detect_patterns(start_point, end_point, window_shift, data_stream, utc_time, plot=1)
        # ==========================================================================================================
        # BCG signal extraction
        movement = band_pass_filtering(data_stream, fs, "bcg")
        # ==========================================================================================================
        # Respiratory signal extraction
        breathing = band_pass_filtering(data_stream, fs, "breath")
        breathing = remove_nonLinear_trend(breathing, 3)
        breathing = savgol_filter(breathing, 11, 3)
        # ==========================================================================================================
        w = modwt(movement, 'bior3.9', 4)
        dc = modwtmra(w, 'bior3.9')
        wavelet_cycle = dc[4]

         # --- HR Estimation via J-peaks (simulated with peak detection) ---
        from detect_peaks import detect_peaks
        peaks = detect_peaks(wavelet_cycle, mpd=fs//2)
        peak_times = pd.to_datetime(rr_data['Timestamp'])
        rr_intervals = np.diff(peak_times.values.astype('datetime64[ms]'))
        rr_intervals = rr_intervals[rr_intervals > np.timedelta64(0, 'ms')]

        # Convert to seconds
        rr_intervals_sec = rr_intervals.astype('timedelta64[ms]').astype('float64') / 1000.0

        # Estimated heart rate from RR intervals
        estimated_hr = 60.0 / rr_intervals_sec

        # Align lengths for comparison
        min_len = min(len(estimated_hr), len(reference_hr))
        estimated_hr = estimated_hr[:min_len]
        reference_hr = rr_data['Heart Rate'].iloc[1:len(estimated_hr)+1].values  # align shapes


        # Error Metrics
        mae = mean_absolute_error(reference_hr, estimated_hr)
        rmse = np.sqrt(mean_squared_error(reference_hr, estimated_hr))
        mape = np.mean(np.abs((reference_hr - estimated_hr) / reference_hr)) * 100

        print('\nHeart Rate Comparison Metrics')
        print('MAE:', mae)
        print('RMSE:', rmse)
        print('MAPE:', mape)

        # Bland-Altman Plot
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

        # Pearson Correlation
        r, _ = pearsonr(reference_hr, estimated_hr)
        plt.figure()
        plt.scatter(reference_hr, estimated_hr)
        plt.title(f"Pearson Correlation (r={r:.2f})")
        plt.xlabel("Reference HR")
        plt.ylabel("Estimated HR")
        plt.savefig('../results/pearson_correlation.png')

        # ==========================================================================================================
        # Vital Signs estimation - (10 seconds window is an optimal size for vital signs measurement)
        t1, t2, window_length, window_shift = 0, 500, 500, 500
        hop_size = math.floor((window_length - 1) / 2)
        limit = int(math.floor(breathing.size / window_shift))
        # ==========================================================================================================
        # Heart Rate
        beats = vitals(t1, t2, window_shift, limit, wavelet_cycle, utc_time, mpd=1, plot=0)
        print('\nHeart Rate Information')
        print('Minimum pulse : ', np.around(np.min(beats)))
        print('Maximum pulse : ', np.around(np.max(beats)))
        print('Average pulse : ', np.around(np.mean(beats)))
        # Breathing Rate
        beats = vitals(t1, t2, window_shift, limit, breathing, utc_time, mpd=1, plot=0)
        print('\nRespiratory Rate Information')
        print('Minimum breathing : ', np.around(np.min(beats)))
        print('Maximum breathing : ', np.around(np.max(beats)))
        print('Average breathing : ', np.around(np.mean(beats)))
        # ==============================================================================================================
        thresh = 0.3
        events = apnea_events(breathing, utc_time, thresh=thresh)
        # ==============================================================================================================
        # Plot Vitals Example
        t1, t2 = 2500, 2500 * 2
        data_subplot(data_stream, movement, breathing, wavelet_cycle, t1, t2)
        # ==============================================================================================================
    print('\nEnd processing ...')
    # ==================================================================================================================