import matplotlib.pyplot as plt
from mef_tools import MefReader
import datetime
import numpy as np
from best.annotations.io import load_CyberPSG
import pytz

# Load the MEF file
path = "data/sub-MH5_ses-EMU1_merged.mefd"
mef = MefReader(path, password2="imagination")

# start_datetime = "19-Jul-2022 23:30:05"
start_datetime = "19-Jul-2022 23:30:10"
end_datetime = "19-Jul-2022 23:30:17"
longer_end_datetime = "19-Jul-2022 23:30:25"

utc = pytz.UTC

start_timestamp = datetime.datetime.strptime(start_datetime, "%d-%b-%Y %H:%M:%S").timestamp() * 1e6
end_timestamp = datetime.datetime.strptime(end_datetime, "%d-%b-%Y %H:%M:%S").timestamp() * 1e6
longer_end_timestamp = datetime.datetime.strptime(longer_end_datetime, "%d-%b-%Y %H:%M:%S").timestamp() * 1e6

# Collect labels
annotations = load_CyberPSG("kyberpejsek.xml")

# In CyberPSG, 07:37 is really 15:37, so we need to apply an offset of 8 hours
start_timestamp += 8 * 60 * 60 * 1e6
end_timestamp += 8 * 60 * 60 * 1e6
longer_end_timestamp += 8 * 60 * 60 * 1e6
start_timestamp = int(start_timestamp)
end_timestamp = int(end_timestamp)
longer_end_timestamp = int(longer_end_timestamp)
total_snapshot_duration = end_timestamp - start_timestamp

# In another dimension of CyberPSG, 23:38 is really 05:28, so we need to apply an offset of 6 hours
annotation_start_time = start_timestamp - 2 * 60 * 60 * 1e6
annotation_end_time = end_timestamp - 2 * 60 * 60 * 1e6
annotation_start = datetime.datetime.fromtimestamp(int(annotation_start_time/1e6))
annotation_end = datetime.datetime.fromtimestamp(int(annotation_end_time/1e6))
annotation_start_dt = utc.localize(annotation_start)
annotation_end_dt = utc.localize(annotation_end)

# Filter annotations to the time range
annotations = annotations[(annotations['start'] >= annotation_start_dt) & (annotations['start'] <= annotation_end_dt)]

# Convert 'start' and 'end' to seconds from the start of the snapshot
annotations['start'] = (annotations['start'] - annotation_start_dt).dt.total_seconds()
annotations['end'] = (annotations['end'] - annotation_start_dt).dt.total_seconds()

print(f"Annotations: {annotations}")

scalp_channel_names = [
    "F8",
    "F9",
    "C3",
    "C4",
    "Cz",
    "O1",
    "O2",
]
diff_channel_names = [
    "TP11",
    "TP12",
]
ieeg_channel_names = [
    "e8-e11",
    "e4-e7",
    "e12-e14",
    "e0-e3",
]

# For each channel, print the datetime of the first and last sample
for channel in scalp_channel_names + diff_channel_names + ieeg_channel_names:
    first_timestamp = mef.get_channel_info(channel)['start_time'][0] / 1e6
    last_timestamp = mef.get_channel_info(channel)['end_time'][0] / 1e6
    first_time = datetime.datetime.fromtimestamp(first_timestamp)
    last_time = datetime.datetime.fromtimestamp(last_timestamp)
    
    # Convert to the format "19-Jul-2022 23:20:24"
    first_time_str = first_time.strftime("%d-%b-%Y %H:%M:%S")
    last_time_str = last_time.strftime("%d-%b-%Y %H:%M:%S")
    print(f"Channel: {channel}, First sample: {first_time_str} ({first_timestamp}), Last sample: {last_time_str} ({last_timestamp})")

# Get all the data
print(f"Querying data from {start_timestamp} to {end_timestamp}")
scalp_data = mef.get_data(scalp_channel_names, start_timestamp, end_timestamp)
diff_data = mef.get_data(diff_channel_names, start_timestamp, end_timestamp)
ieeg_data = np.array(mef.get_data(ieeg_channel_names, start_timestamp, end_timestamp))

# Get one longer data sample (from e8-e11)
longer_sample = mef.get_data(ieeg_channel_names[0], start_timestamp, longer_end_timestamp)
# Export the longer sample to a PDF
fig, axs = plt.subplots(2, 1, figsize=(12, 4), gridspec_kw={'hspace': 0.1})

# Full signal
axs[0].plot(longer_sample, color='black', linewidth=0.5, alpha=0.3)
# Highlight the zoomed segment in the upper plot
start_idx = int(len(longer_sample) * 0.33)
end_idx = int(len(longer_sample) * 0.41)
axs[0].plot(range(start_idx, end_idx), longer_sample[start_idx:end_idx], color='black', linewidth=1.0, alpha=0.6)
axs[0].axis('off')  # Turn off all axes, ticks, labels
# Print how long in seconds the zoomed segment is
print(f"Zoomed segment duration: {(end_idx - start_idx) / 250} seconds")

# Zoomed segment (33% to 41%)
zoomed_segment = longer_sample[start_idx:end_idx]
axs[1].plot(range(start_idx, end_idx), zoomed_segment, color='black', linewidth=1, alpha=1.0)
axs[1].axis('off')

plt.savefig("longer_sample_e8-e11.pdf", bbox_inches='tight', pad_inches=0)

# Calculate scalp diff data
diff_data_ordered = np.array(
    [
        diff_data[0],
        diff_data[1],
        diff_data[1],
        diff_data[0],
        diff_data[1],
        diff_data[1],
        diff_data[0],
    ]
)
scalp_data = scalp_data - diff_data_ordered
scalp_channel_names = [
    'F8 - TP11',
    'F9 - TP12',
    'C3 - TP12',
    'C4 - TP11',
    'Cz - TP12',
    'O1 - TP12',
    'O2 - TP11',
]

# Print shapes
print(scalp_data.shape)  # 7 channels
print(ieeg_data.shape)   # 4 channels

# Calculate sampling rates for each channel
scalp_sampling_rates = [len(scalp_data[i]) / (total_snapshot_duration / 1e6) for i in range(len(scalp_channel_names))]
ieeg_sampling_rates = [len(ieeg_data[i]) / (total_snapshot_duration / 1e6) for i in range(len(ieeg_channel_names))]

# Convert annotation start and end times to X values for each channel
annotations['start_x'] = annotations['start']
annotations['end_x'] = annotations['end']

# Print min and max values for each channel
for i, channel in enumerate(scalp_channel_names):
    print(f"{channel}: Min: {np.min(scalp_data[i])}, Max: {np.max(scalp_data[i])}")
for i, channel in enumerate(ieeg_channel_names):
    print(f"{channel}: Min: {np.min(ieeg_data[i])}, Max: {np.max(ieeg_data[i])}")

# Plot scalp data above, ieeg data below, to PDF, lineplots with enhanced visuals
fig, ax = plt.subplots(11, 1, figsize=(24, 24), gridspec_kw={'hspace': 0.1, 'wspace': 0})  # Add space between plots
for i, channel in enumerate(scalp_channel_names):
    ax[i].plot(scalp_data[i], color='black', linewidth=1, alpha=0.35)  # Scalp signals in light gray
    channel_annotations = annotations[annotations['channel'] == channel]
    for _, annotation in channel_annotations.iterrows():
        if annotation['annotation'] == 'Label':
            ax[i].plot(
                range(int(annotation['start_x'] * scalp_sampling_rates[i]), int(annotation['end_x'] * scalp_sampling_rates[i])),
                scalp_data[i][int(annotation['start_x'] * scalp_sampling_rates[i]):int(annotation['end_x'] * scalp_sampling_rates[i])],
                color='black', linewidth=2.0, alpha=1.0  # Labeled regions in black and thicker
            )
    ax[i].set_ylim([-50, 50])  # Set y-axis limits for scalp plots
    ax[i].axis('off')
for i, channel in enumerate(ieeg_channel_names):
    if i == 0:
        ylo = -210
        yhi = -50
    elif i == 1:
        ylo = -600
        yhi = 200
    elif i == 2:
        ylo = -500
        yhi = 300
    elif i == 3:
        ylo = -270
        yhi = -110
    
    ax[i + 7].set_ylim([ylo, yhi])  # Set y-axis limits for iEEG plots
    ax[i + 7].plot(ieeg_data[i], color='black', linewidth=1, alpha=0.45)  # iEEG signals in gray
    channel_annotations = annotations[annotations['channel'] == channel]
    for _, annotation in channel_annotations.iterrows():
        if annotation['annotation'] == 'Label':
            ax[i + 7].plot(
                range(int(annotation['start_x'] * ieeg_sampling_rates[i]), int(annotation['end_x'] * ieeg_sampling_rates[i])),
                ieeg_data[i][int(annotation['start_x'] * ieeg_sampling_rates[i]):int(annotation['end_x'] * ieeg_sampling_rates[i])],
                color='black', linewidth=2.0  # Labeled regions in black and thicker
            )
        elif annotation['annotation'] == 'Prediction':
            start_x = annotation['start_x'] * ieeg_sampling_rates[i]
            end_x = annotation['end_x'] * ieeg_sampling_rates[i]
            ax[i + 7].add_patch(
                plt.Rectangle(
                    (start_x, min(ieeg_data[i])),  # Bottom-left corner
                    end_x - start_x,  # Width
                    max(ieeg_data[i]) - min(ieeg_data[i]),  # Height
                    color='red',
                    alpha=0.3
                )
            )
    ax[i + 7].axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)  # Adjust spacing
plt.savefig("scalp_ieeg_data_raw.pdf", bbox_inches='tight', pad_inches=0)

# Find all labeled spindles for the e8-e11 channel
labeled_spindles = annotations[(annotations['channel'] == 'e8-e11') & (annotations['annotation'] == 'Label')]
if len(labeled_spindles) < 2:
    raise ValueError("Less than two labeled spindles found for channel e8-e11.")

# Reduce padding size
padding = 0.2  # seconds

# Convert spindle start and end times to sample indices
spindle_samples = []
for _, spindle in labeled_spindles.iterrows():
    spindle_start_sample = int(spindle['start_x'] * ieeg_sampling_rates[0])
    spindle_end_sample = int(spindle['end_x'] * ieeg_sampling_rates[0])
    spindle_samples.append((spindle_start_sample, spindle_end_sample))

# Calculate zoomed-in range for the second spindle with reduced padding
second_spindle = labeled_spindles.iloc[1]
start_x_zoom = max(0, second_spindle['start_x'] - padding)
end_x_zoom = min(len(ieeg_data[0]) / ieeg_sampling_rates[0], second_spindle['end_x'] + padding)
start_sample = int(start_x_zoom * ieeg_sampling_rates[0])
end_sample = int(end_x_zoom * ieeg_sampling_rates[0])

# Clip the full plot to start after the first spindle
first_spindle_end_sample = spindle_samples[0][1]  # End of the first spindle
clipped_ieeg_data = ieeg_data[0][first_spindle_end_sample:]  # Clip the signal
clipped_start_sample = first_spindle_end_sample  # Adjust start sample for plotting

# Plot the e8-e11 channel and zoomed-in spindle
fig, ax = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'hspace': 0.5})

# Full e8-e11 channel (clipped)
ax[0].plot(
    range(clipped_start_sample, clipped_start_sample + len(clipped_ieeg_data)),
    clipped_ieeg_data,
    color='black', linewidth=1.5, alpha=0.45  # Thicker signal
)
for spindle_start_sample, spindle_end_sample in spindle_samples[1:]:  # Highlight only spindles after the first
    ax[0].plot(
        range(spindle_start_sample, spindle_end_sample),
        ieeg_data[0][spindle_start_sample:spindle_end_sample],
        color='black', linewidth=2.0, alpha=1.0  # Highlight in black
    )
ax[0].set_ylim([-210, -50])  # Set y-axis limits for consistency
ax[0].set_title("Channel e8-e11 (Clipped)")
ax[0].axis('off')

# Zoomed-in spindle
ax[1].plot(
    range(start_sample, end_sample),
    ieeg_data[0][start_sample:end_sample],
    color='black', linewidth=2.0, alpha=0.45  # Thicker padding
)
ax[1].plot(
    range(spindle_start_sample, spindle_end_sample),
    ieeg_data[0][spindle_start_sample:spindle_end_sample],
    color='black', linewidth=4.0, alpha=1.0  # Thicker highlight for the spindle
)
ax[1].set_ylim([-210, -50])  # Set y-axis limits for consistency
ax[1].set_title("Zoomed-in View of Second Labeled Spindle")
ax[1].axis('off')

# Save the plot
plt.savefig("e8_e11_zoomed_spindle.pdf", bbox_inches='tight', pad_inches=0)

