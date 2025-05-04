import pandas as pd

# Input and output file paths
input_file = 'data/BCG/02_20231103_BCG.csv'     # your original file
output_file = 'data/BCG/fixed_bcg.csv'     # new clean file

# Read the first 2 lines manually
with open(input_file, 'r') as f:
    header = f.readline().strip().split(',')
    first_row = f.readline().strip().split(',')
    rest_data = f.readlines()

# Get initial values
initial_bcg = int(first_row[0])
initial_timestamp = int(first_row[1])
fs = int(first_row[2])  # sampling rate (e.g. 140 Hz)
period_ms = int(1000 / fs)

# Start filling
bcg_values = [initial_bcg]
timestamps = [initial_timestamp]
sampling_rates = [fs]

# Process the remaining lines
for i, line in enumerate(rest_data):
    value = line.strip()
    if value == '':
        continue
    bcg_values.append(int(value))
    timestamps.append(initial_timestamp + (i + 1) * period_ms)
    sampling_rates.append(fs)

# Create final DataFrame
fixed_df = pd.DataFrame({
    'BCG': bcg_values,
    'Timestamp': timestamps,
    'fs': sampling_rates
})

# Save to new CSV
fixed_df.to_csv(output_file, index=False)

print(f"✅ Fixed file saved to: {output_file}")
print(f"🧾 Total samples: {len(fixed_df)}")
