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
from scipy import signal

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

file = 'data/BCG/fixed_bcg2.csv'
reference_rr_file = 'data/RR/03_20231105_RR.csv'

if file.endswith(".csv"):
    fileName = os.path.join(file)
    if os.stat(fileName).st_size != 0:
        rawData = pd.read_csv(fileName)
        utc_time = rawData["Timestamp"].values
        print("\U0001F4C5 BCG Start:", pd.to_datetime(utc_time.min(), unit='ms'))
        print("\U0001F4C5 BCG End  :", pd.to_datetime(utc_time.max(), unit='ms'))
        data_stream = rawData["BCG"].values
        fs_original = int(rawData["fs"].iloc[0])
        print("✅ Loaded samples:", len(data_stream))
        print("✅ Sampling frequency (Hz):", 50 )

        # Load ECG-declived reference HR
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
        
        # Resample BCG to 50Hz
        # n_samples_50hz = int((end_time - start_time) / 1000 * fs)
        # synced_bcg_resampled = resample(synced_bcg, n_samples_50hz)
        # N = len(synced_bcg_resampled)
        

        # Step 1: Compute the true time-based duration of the synced BCG signal
        start_time = pd.to_datetime(synced_utc[0], unit='ms')
        end_time = pd.to_datetime(synced_utc[-1], unit='ms')
        duration_sec = (end_time - start_time).total_seconds()

        # Step 2: Calculate resampled size based on time, not sample count
        fs_target = 50
        num_samples = int(duration_sec * fs_target)

        # Step 3: Resample BCG signal
        resampled_bcg = resample(synced_bcg, num_samples)

        # Step 4: Generate accurate time index for 50Hz spacing
        resampled_index = pd.date_range(start=start_time, periods=num_samples, freq='20ms')

        # Step 5: Create DataFrame
        resampled_df = pd.DataFrame({
            'Timestamp': resampled_index,
            'BCG': resampled_bcg
        })
        resampled_df.set_index('Timestamp', inplace=True)

        # Step 6: Time axis in milliseconds for later processing
        resampled_time = np.arange(0, len(resampled_df) * 20, 20)

        # Debug print
        print("⏱ Duration (seconds):", duration_sec)
        print("🧠 Resampled BCG samples:", len(resampled_df))
        print("\U0001F552 Sync window (ms):", resampled_index[0], "→", resampled_index[-1])

        # Denoise and remove movement artifacts
        # Denoise and remove movement artifacts
        start_point, end_point, window_shift = 0, 500, 500
        clean_bcg, clean_time = detect_patterns(
            start_point,
            end_point,
            window_shift,
            resampled_df['BCG'].values,
            resampled_time,
            plot=1
        )

        # Filter for HR and respiratory components
        movement = band_pass_filtering(clean_bcg, fs_target, "bcg")
        breathing = band_pass_filtering(clean_bcg, fs_target, "breath")
        breathing = remove_nonLinear_trend(breathing, 3)
        breathing = savgol_filter(breathing, 11, 3)

        # Wavelet transform
        w = modwt(movement, 'bior3.9', 4)
        dc = modwtmra(w, 'bior3.9')
        wavelet_cycle = dc[4]  # Level 4 smooth
        

        #plot wavelet cycle
        plt.figure()
        plt.plot(clean_time, wavelet_cycle, color='k', lw=2)
        plt.title('Wavelet Cycle')
        plt.xlabel('Time [ms]')
        plt.ylabel('Amplitude')
        #save figure
        plt.savefig('results/wavelet_cycle.png')


        # Peak detection from filtered BCG
        peaks = detect_peaks(wavelet_cycle)
        peak_times = clean_time[peaks] / 1000.0  # ms to seconds
        rr_intervals_sec = np.diff(peak_times)
        estimated_hr = 60.0 / rr_intervals_sec


        # Compute timestamps for estimated HR (at midpoints between peaks)
        estimated_hr_times = peak_times[1:]  # aligns each HR value with the second peak

        # Use raw RR values directly (only for plotting and comparison, not interpolation)
        reference_hr = synced_rr['Heart Rate'].replace(0, np.nan).dropna().values

        # Truncate estimated HR to match length
        min_len = min(len(reference_hr), len(estimated_hr))
        estimated_hr = estimated_hr[:min_len]
        reference_hr = reference_hr[:min_len]
        estimated_hr_times = estimated_hr_times[:min_len]


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
        plt.savefig('results/bland_altman.png')

        # Pearson
        r, _ = pearsonr(reference_hr, estimated_hr)
        print(f"r value is {r}")
        plt.figure()
        plt.scatter(reference_hr, estimated_hr)
        plt.title(f"Pearson Correlation (r={r:.2f})")
        plt.xlabel("Reference HR")
        plt.ylabel("Estimated HR")
        plt.savefig('results/pearson_correlation.png')

        # Vitals
        t1, t2, window_length, window_shift = 0, 500, 500, 500
        hop_size = math.floor((window_length - 1) / 2)
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

        # events = apnea_events(breathing, clean_time, thresh=0.3)

        # Visualization
        t1, t2 = 2500, 5000
        data_subplot(clean_bcg, movement, breathing, wavelet_cycle, t1, t2)
        
        plt.figure()
        plt.plot(reference_hr, label="Reference HR")
        plt.plot(estimated_hr, label="Estimated HR")
        plt.legend()
        plt.title("Comparison of Heart Rates")
        plt.xlabel("Index")
        plt.ylabel("Heart Rate (BPM)")
        plt.savefig('results/hr_comparison.png')
        print("Reference HR stats:", np.min(reference_hr), np.max(reference_hr), np.mean(reference_hr))
        print("Estimated HR stats:", np.min(estimated_hr), np.max(estimated_hr), np.mean(estimated_hr))
        
        # num_windows = (N - start_point) // window_shift
        # print(f"Signal length: {N}, Expected number of windows: {num_windows}")



                # Save

        # Save summary results to file
        summary_text = f"""Heart Rate Comparison Metrics
        MAE: {mae}
        RMSE: {rmse}
        MAPE: {mape}
        r value is {r}

        Heart Rate Information
        Minimum pulse :  {np.around(np.min(hr_beats))}
        Maximum pulse :  {np.around(np.max(hr_beats))}
        Average pulse :  {np.around(np.mean(hr_beats))}

        Respiratory Rate Information
        Minimum breathing :  {np.around(np.min(breath_beats))}
        Maximum breathing :  {np.around(np.max(breath_beats))}
        Average breathing :  {np.around(np.mean(breath_beats))}

        Reference HR stats: {np.min(reference_hr)} {np.max(reference_hr)} {np.mean(reference_hr)}
        Estimated HR stats: {np.min(estimated_hr)} {np.max(estimated_hr)} {np.mean(estimated_hr)}
        """

        # Ensure the directory exists
        os.makedirs('results', exist_ok=True)

        # Write to file
        with open('results/Output.py', 'w') as f:
            f.write(summary_text)

    print('\nEnd processing ...')