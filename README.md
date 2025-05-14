
# BCG-HeartRateDetection

A Python-based pipeline for heart rate detection from bed-embedded Ballistocardiogram (BCG) sensors. This system processes raw BCG signals, filters out noise and movement artifacts, detects peaks corresponding to heartbeats, and estimates heart rate over time. The results are compared against ECG-derived reference heart rates for evaluation.

## 📁 Project Structure
```bash

data/
├── BCG/
│ └── fixed_bcg2.csv # Raw BCG data file
├── RR/
│ └── 03_20231105_RR.csv # Reference heart rate (ECG-based RR intervals)

results/
├── wavelet_cycle.png # Visualization of wavelet-smoothed BCG signal
├── hr_comparison.png # Reference vs Estimated HR
├── bland_altman.png # Bland-Altman plot
├── pearson_correlation.png # Correlation between estimated and reference HR

````

## ⚙️ Pipeline Overview

1. **Load BCG and Reference RR Data**
2. **Synchronize BCG and RR Timeframes**
3. **Resample BCG to 50 Hz**
4. **Denoise and Remove Artifacts**
   - Band-pass filtering
   - Non-linear trend removal
   - Smoothing with Savitzky-Golay filter
5. **Wavelet Transform**
   - Multi-resolution analysis (MODWT)
6. **Peak Detection**
   - Detect peaks in smoothed BCG signal to derive heartbeats
7. **Heart Rate Estimation**
   - Calculate RR intervals and convert to HR in BPM
8. **Metrics Evaluation**
   - MAE, RMSE, MAPE, Pearson correlation
   - Comparison against reference ECG-derived HR
9. **Visualization**

## 📈 Example Output

```bash
Heart Rate Comparison Metrics
MAE: 16.08
RMSE: 22.65
MAPE: 19.14%
r value: ~0.00

Heart Rate Information
Min: 53 bpm
Max: 91 bpm
Mean: 72 bpm
````

## 📦 Dependencies

Make sure to install the following Python packages:

```bash
pip install numpy pandas scipy matplotlib scikit-learn
```

## 📌 Notes

- Data is resampled to 50 Hz using linear interpolation and `scipy.signal.resample`.
- Reference HR is used for validation only and is not interpolated.
- Ensure your CSV files are timestamped in milliseconds and aligned properly.
